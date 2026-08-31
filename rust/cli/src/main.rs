//! msspeculator Rust inference CLI: FASTA libraries, peptide prediction, and model diagnostics.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::Result;
use clap::{Parser, Subcommand};
use msspeculator_core::{
    builtin, fit, predict, speclib, BuiltinModel, ModelSource, MsContext, Prediction,
};
use serde_json::json;

mod diagnostics;
mod progress;
use msspeculator_inference::library;

#[derive(Parser)]
#[command(
    name = "msspeculator-cli",
    about = "Generate libraries, predict peptides, or render model diagnostics."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Predict one peptide and print one JSON object.
    Predict(PredictArgs),
    /// Digest a FASTA and write a DIA-NN TSV spectral library.
    Library(LibraryArgs),
    /// Run a built-in model health panel and write diagnostic artifacts.
    RunDoctor(DoctorArgs),
    /// Fit the acquisition context that best explains a published spectral library.
    FitContext(FitContextArgs),
}

/// `--add-unimod ACCESSION[:MASS]`: a modification the library contains.
///
/// The mass is optional and only needed when the file spells a shift more coarsely than the
/// automatic tolerance accepts, DIA-NN writes 6C-CysPAT as `+221.082`, which is 3e-4 from the
/// table. Stating it is a declaration that the rounding is intended, not a widening of the
/// tolerance for every modification.
#[derive(Clone)]
struct AddUnimod {
    accession: u32,
    observed_mass: Option<f64>,
}

impl FromStr for AddUnimod {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (accession, mass) = match value.split_once(':') {
            Some((accession, mass)) => (accession, Some(mass)),
            None => (value, None),
        };
        Ok(Self {
            accession: accession
                .trim()
                .parse()
                .map_err(|_| format!("{accession:?} is not a UNIMOD accession"))?,
            observed_mass: match mass {
                Some(mass) => Some(
                    mass.trim()
                        .parse()
                        .map_err(|_| format!("{mass:?} is not a mass"))?,
                ),
                None => None,
            },
        })
    }
}

#[derive(clap::Args)]
struct ArtifactArgs {
    /// The model: a path to a .safetensors artifact (from `msspeculator export-rust`), or
    /// `builtin:NAME` for one compiled into this binary. Defaults to the bundled model, so a
    /// fresh build predicts without staging anything.
    #[arg(long, default_value = "builtin:small-v0")]
    model: String,
    /// Override the artifact activation for a controlled inference benchmark.
    #[arg(long, value_name = "ACTIVATION")]
    activation: Option<String>,
}

/// A parsed `--ms-context`, keeping the text the user wrote so the report echoes their words
/// rather than a re-rendering of them.
#[derive(Clone)]
struct MsContextArg {
    raw: String,
    resolved: MsContext,
}

impl FromStr for MsContextArg {
    type Err = String;

    /// Four `::`-separated acquisition factors, or a bare name for a setup fitted into the
    /// artifact. The separator is what distinguishes them: a library's setup name is a label,
    /// and nothing about it parses as a factor list, so there is no ambiguity to resolve.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let raw = value.to_string();
        if !value.contains("::") {
            if value.trim().is_empty() {
                return Err(
                    "expected a setup name or INSTRUMENT::DETECTOR::FRAGMENTATION::ENERGY"
                        .to_string(),
                );
            }
            return Ok(Self {
                raw,
                resolved: MsContext::Named(value.to_string()),
            });
        }
        let parts: Vec<&str> = value.split("::").collect();
        if parts.len() != 4 {
            return Err("expected INSTRUMENT::DETECTOR::FRAGMENTATION::ENERGY".to_string());
        }
        let energy = parts[3]
            .parse()
            .map_err(|_| format!("energy {:?} is not a number", parts[3]))?;
        Ok(Self {
            raw,
            resolved: MsContext::Factors {
                instrument: parts[0].to_string(),
                detector: parts[1].to_string(),
                fragmentation: parts[2].to_string(),
                energy: Some(energy),
            },
        })
    }
}

#[derive(clap::Args)]
struct ContextArgs {
    /// Acquisition context: a named setup fitted with `fit-context`, or the factors spelled out
    /// as "INSTRUMENT::DETECTOR::FRAGMENTATION::ENERGY".
    #[arg(long, conflicts_with = "nce")]
    ms_context: Option<MsContextArg>,
    /// Shorthand: collision energy only (unknown instrument/detector/fragmentation).
    #[arg(long, conflicts_with = "ms_context")]
    nce: Option<f32>,
    /// Named chromatography context (dataset) for raw RT; absent means context-free iRT.
    #[arg(long)]
    chrom_context: Option<String>,
}

impl ContextArgs {
    fn ms_context(&self) -> Option<MsContext> {
        self.ms_context
            .as_ref()
            .map(|context| context.resolved.clone())
            .or_else(|| {
                self.nce.map(|energy| MsContext::Factors {
                    instrument: String::new(),
                    detector: String::new(),
                    fragmentation: String::new(),
                    energy: Some(energy),
                })
            })
    }
}

#[derive(clap::Args)]
struct PredictArgs {
    #[command(flatten)]
    artifact: ArtifactArgs,
    /// Peptide in the supported ProForma subset, for example `PEPC[UNIMOD:4]IDER`.
    #[arg(long)]
    peptide: String,
    /// Precursor charge.
    #[arg(long)]
    charge: i64,
    #[command(flatten)]
    context: ContextArgs,
    /// Drop fragments below this base-peak-relative intensity.
    #[arg(long, default_value_t = 0.01)]
    min_intensity: f64,
}

#[derive(clap::Args)]
struct LibraryArgs {
    #[command(flatten)]
    artifact: ArtifactArgs,
    /// FASTA to digest.
    #[arg(long)]
    fasta: PathBuf,
    /// Output path. The suffix picks the format: `.mzspeclib.txt` writes mzSpecLib, which carries
    /// its own provenance, and anything else writes DIA-NN TSV. A trailing `.gz` compresses either.
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 2)]
    missed_cleavages: usize,
    #[arg(long, default_value_t = 7)]
    min_length: usize,
    #[arg(long, default_value_t = 30)]
    max_length: usize,
    #[arg(long, default_value_t = 2)]
    min_charge: i64,
    #[arg(long, default_value_t = 4)]
    max_charge: i64,
    /// Fixed modification rule. Repeat to add rules; defaults to `C[UNIMOD:4]`.
    #[arg(long, value_name = "TARGETS[MOD]", action = clap::ArgAction::Append)]
    fixed_mod: Vec<String>,
    /// Disable the default fixed `C[UNIMOD:4]` rule.
    #[arg(long, conflicts_with = "fixed_mod")]
    no_fixed_mods: bool,
    /// Variable modification rule. Repeat to add rules; defaults to `M[UNIMOD:35]`.
    #[arg(long, value_name = "TARGETS[MOD]", action = clap::ArgAction::Append)]
    variable_mod: Vec<String>,
    /// Maximum total number of variable modification placements per peptide.
    #[arg(long, default_value_t = 1)]
    max_variable_mods: usize,
    #[command(flatten)]
    context: ContextArgs,
    /// Drop fragments below this base-peak-relative intensity.
    #[arg(long, default_value_t = 0.01)]
    min_intensity: f64,
    /// Keep at most this many of the strongest fragments per precursor. Applied after
    /// `--min-intensity`, so a precursor with fewer surviving peaks keeps all of them.
    #[arg(long, value_name = "N")]
    max_fragments: Option<usize>,
    /// Where to write the resolved-configuration sidecar. Defaults to `<out>.config.json`.
    #[arg(long, value_name = "PATH", conflicts_with = "no_config_out")]
    config_out: Option<PathBuf>,
    /// Skip the resolved-configuration sidecar.
    #[arg(long)]
    no_config_out: bool,
    /// Add pseudo-reversed decoy precursors. Decoys colliding with target sequences are skipped.
    #[arg(long)]
    decoys: bool,
    /// Suppress the progress line. It already goes quiet when stderr is not a terminal, so this
    /// is for a log that should carry only the summary.
    #[arg(long)]
    no_progress: bool,
}

#[derive(clap::Args)]
struct DoctorArgs {
    #[command(flatten)]
    artifact: ArtifactArgs,
    /// Directory for diagnostic artifacts.
    #[arg(long, default_value = "model-doctor")]
    out: PathBuf,
}

#[derive(clap::Args)]
struct FitContextArgs {
    #[command(flatten)]
    artifact: ArtifactArgs,
    /// Local path to a Spectronaut or DIA-NN TSV library. Remote libraries are downloaded first:
    /// this reads the file directly and has no object-store client.
    #[arg(long)]
    library: PathBuf,
    /// Modification present in the library, as ACCESSION or ACCESSION:MASS. Repeatable.
    #[arg(long = "add-unimod", value_name = "ACCESSION[:MASS]")]
    add_unimod: Vec<AddUnimod>,
    /// Acquisition factors the library was measured with. Omitted names resolve to the neutral
    /// row, so a library from an unrecorded setup still fits.
    #[arg(long, default_value = "")]
    instrument: String,
    #[arg(long, default_value = "")]
    detector: String,
    #[arg(long, default_value = "")]
    fragmentation: String,
    /// Maximum passes over the training precursors. Held-out agreement is checked after each and
    /// the best-scoring context is returned, so raising this cannot make the answer worse.
    #[arg(long, default_value_t = 12)]
    epochs: usize,
    /// Act on DIA-NN's `ExcludeFromAssay`. Off by default: it marks transitions skipped for
    /// quantification, not wrong ones, and honouring it can cut a library's depth by two thirds.
    #[arg(long)]
    drop_excluded: bool,
    /// Store the fitted context in the artifact under this name, addressable afterwards as
    /// `--ms-context NAME`. Requires `--out`; refitting an existing name replaces its row.
    #[arg(long, value_name = "NAME", requires = "out")]
    save_as: Option<String>,
    /// Where to write the artifact carrying the fitted row. Never in place: the input artifact
    /// is the reference point a fit is judged against.
    #[arg(long, requires = "save_as")]
    out: Option<PathBuf>,
}

fn run_fit_context(args: FitContextArgs) -> Result<()> {
    let mut artifact = builtin::load_model(&args.artifact.model)?.artifact;
    library::apply_activation_override(&mut artifact, args.artifact.activation.as_deref())?;

    let spec = speclib::LibrarySpec {
        context: args.library.display().to_string(),
        instrument: args.instrument.clone(),
        detector: args.detector.clone(),
        fragmentation: args.fragmentation.clone(),
        aliases: args
            .add_unimod
            .iter()
            .map(|declared| speclib::ModAlias {
                accession: declared.accession,
                observed_mass: declared.observed_mass,
            })
            .collect(),
        retention: speclib::RetentionSource::Normalized,
        drop_excluded: args.drop_excluded,
    };
    let (precursors, stats) = speclib::read_speclib(&args.library, &spec)?;
    if !stats.unmapped_masses.is_empty() {
        anyhow::bail!(
            "library contains mass shifts no --add-unimod explains: {}. Declare each one, with \
             its spelled mass if the file rounds it.",
            stats.unmapped_masses.join(", ")
        );
    }

    let config = fit::FitConfig {
        epochs: args.epochs,
        ..fit::FitConfig::default()
    };
    let report = fit::fit_ms_context(&artifact, &precursors, &config)?;
    let mut saved = serde_json::Value::Null;
    if let (Some(name), Some(out)) = (&args.save_as, &args.out) {
        let row = artifact.set_ms_context(name, &report.context)?;
        artifact.save(out)?;
        saved = serde_json::json!({"setup": name, "row": row, "artifact": out});
    }
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "library": args.library.display().to_string(),
            "stats": {
                "rows": stats.rows,
                "decoys": stats.decoys,
                "excluded": stats.excluded,
                "precursors": stats.precursors,
                "precursors_without_fragments": stats.precursors_without_fragments,
                "fragments_dropped": stats.fragments_dropped,
            },
            "split": {
                "salt": config.split.salt,
                "train": report.train,
                "val": report.val,
                "test": report.test,
            },
            "fit": {
                "context_dim": report.context_dim,
                "spectral_angle_before": report.spectral_angle_before,
                "spectral_angle_after": report.spectral_angle_after,
                "context": report.context,
            },
            "objective": report.objective,
            "held_out": report.held_out,
            "saved": saved,
        }))?
    );
    Ok(())
}

fn to_json(
    prediction: &Prediction,
    ms_context: Option<&String>,
    chrom_context: Option<&String>,
) -> serde_json::Value {
    json!({
        "peptide": prediction.peptide,
        "charge": prediction.charge,
        "precursor_mz": prediction.precursor_mz,
        "rt": prediction.rt,
        // Only present with a chromatography context, where `rt` is that dataset's gradient time
        // and this is the context-free index the same peptide sits at.
        "irt": prediction.irt,
        "ccs": prediction.ccs,
        "ms_context": ms_context,
        "chrom_context": chrom_context,
        "fragments": {
            "ion": prediction.fragments.ion,
            "ord": prediction.fragments.ord,
            "z": prediction.fragments.z,
            "mz": prediction.fragments.mz,
            "rel": prediction.fragments.rel,
        }
    })
}

fn run_predict(args: PredictArgs) -> Result<()> {
    let mut artifact = builtin::load_model(&args.artifact.model)?.artifact;
    library::apply_activation_override(&mut artifact, args.artifact.activation.as_deref())?;
    let ms_context = args.context.ms_context();
    let prediction = predict(
        &artifact,
        &args.peptide,
        args.charge,
        ms_context.as_ref(),
        args.context.chrom_context.as_deref(),
        args.min_intensity,
    )?;
    let raw_ms_context = args.context.ms_context.as_ref().map(|context| &context.raw);
    println!(
        "{}",
        serde_json::to_string(&to_json(
            &prediction,
            raw_ms_context,
            args.context.chrom_context.as_ref(),
        ))?
    );
    Ok(())
}

fn parse_model_source(spec: &str) -> Result<ModelSource> {
    match spec.strip_prefix(builtin::BUILTIN_PREFIX) {
        Some("small-v0") => Ok(ModelSource::Builtin(BuiltinModel::SmallV0)),
        Some(name) => anyhow::bail!(
            "unknown builtin model {name:?}; this build carries: {}",
            builtin::names().join(", ")
        ),
        None => Ok(ModelSource::File(PathBuf::from(spec))),
    }
}

/// Append to a path's whole name rather than replacing its extension.
///
/// `lib.mzspeclib.txt` gains `.config.json` to become `lib.mzspeclib.txt.config.json`;
/// `Path::with_extension` would have produced `lib.mzspeclib.config.json` and lost which format
/// the sidecar describes. Built as an OS string, since a path need not be UTF-8.
fn append_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.to_path_buf().into_os_string();
    name.push(suffix);
    PathBuf::from(name)
}

impl LibraryArgs {
    /// Written by default: a library whose settings live only in a shell history cannot be
    /// regenerated. `--no-config-out` is the explicit opt out, and clap already refuses it
    /// together with `--config-out`.
    fn sidecar_path(&self) -> Option<PathBuf> {
        if self.no_config_out {
            return None;
        }
        Some(
            self.config_out
                .clone()
                .unwrap_or_else(|| append_suffix(&self.out, ".config.json")),
        )
    }
}

fn run_library(args: LibraryArgs) -> Result<()> {
    let ms_context = args.context.ms_context();
    let default_fixed = ["C[UNIMOD:4]".to_string()];
    let default_variable = ["M[UNIMOD:35]".to_string()];
    let fixed_mods: &[String] = if args.no_fixed_mods {
        &[]
    } else if args.fixed_mod.is_empty() {
        &default_fixed
    } else {
        &args.fixed_mod
    };
    let variable_mods: &[String] = if args.variable_mod.is_empty() {
        &default_variable
    } else {
        &args.variable_mod
    };
    let config_out = args.sidecar_path();
    let line = progress::ProgressLine::for_stderr(!args.no_progress);
    let report = |progress| line.update(progress);
    let stats = library::write_library(&library::LibraryOptions {
        out: &args.out,
        config_out: config_out.as_deref(),
        stream: library::StreamOptions {
            model: parse_model_source(&args.artifact.model)?,
            fasta: &args.fasta,
            activation: args.artifact.activation.as_deref(),
            ms_context: ms_context.as_ref(),
            chrom_context: args.context.chrom_context.as_deref(),
            min_intensity: args.min_intensity,
            missed_cleavages: args.missed_cleavages,
            min_length: args.min_length,
            max_length: args.max_length,
            min_charge: args.min_charge,
            max_charge: args.max_charge,
            fixed_mods,
            variable_mods,
            max_variable_mods: args.max_variable_mods,
            max_fragments: args.max_fragments,
            generate_decoys: args.decoys,
            progress: Some(&report),
        },
    })?;
    // No explicit wipe: `ProgressLine`'s `Drop` restores the terminal on the error path too,
    // which is the path that has it drawn.
    drop(line);
    eprintln!(
        "{} proteins -> {} peptides -> {} precursors ({} decoys) -> {} fragments -> {} in {:.1}s ({:.1}s digesting)",
        stats.proteins,
        stats.peptides,
        stats.precursors,
        stats.decoys,
        stats.fragments,
        args.out.display(),
        (stats.digest + stats.predict).as_secs_f64(),
        stats.digest.as_secs_f64(),
    );
    Ok(())
}

fn run_doctor(args: DoctorArgs) -> Result<()> {
    let mut artifact = builtin::load_model(&args.artifact.model)?.artifact;
    library::apply_activation_override(&mut artifact, args.artifact.activation.as_deref())?;
    let report = diagnostics::run_doctor(&artifact, &args.out)?;
    println!("{}", report.terminal_plot);
    eprintln!(
        "{} iRT standards -> {}, {}, {} (slope={:.4}, intercept={:.4}, R2={:.4}, MAE={:.4})",
        report.summary.n,
        report.svg_path.display(),
        report.report_path.display(),
        report.predictions_path.display(),
        report.summary.slope,
        report.summary.intercept,
        report.summary.r_squared,
        report.summary.mae
    );
    // Retention says whether the model is on the right scale; this says whether it predicts the
    // spectrum, which is the part a search engine actually consumes.
    for score in &report.spectra {
        eprintln!(
            "{:<26} {:<26} z={} spectral_angle={:.4}",
            score.dataset, score.proforma, score.charge, score.spectral_angle
        );
    }
    eprintln!(
        "{} reference spectra -> mean spectral angle {:.4}",
        report.spectra.len(),
        report.mean_spectral_angle()
    );
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Predict(args) => run_predict(args),
        Command::Library(args) => run_library(args),
        Command::RunDoctor(args) => run_doctor(args),
        Command::FitContext(args) => run_fit_context(args),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `.config.json` appends to the whole name, so which format the sidecar describes stays
    /// visible and the two files sort next to each other.
    #[test]
    fn the_sidecar_defaults_beside_the_library_it_describes() {
        for (out, expected) in [
            ("lib.tsv", "lib.tsv.config.json"),
            ("lib.mzspeclib.txt", "lib.mzspeclib.txt.config.json"),
            ("lib.mzspeclib.txt.gz", "lib.mzspeclib.txt.gz.config.json"),
            ("no-extension", "no-extension.config.json"),
        ] {
            assert_eq!(
                append_suffix(Path::new(out), ".config.json"),
                PathBuf::from(expected),
                "{out}"
            );
        }
    }

    #[test]
    fn ms_context_reads_a_bare_name_as_a_setup_and_four_parts_as_factors() {
        let factors: MsContextArg = "Lumos::FTMS::HCD::28".parse().unwrap();
        match factors.resolved {
            MsContext::Factors {
                instrument, energy, ..
            } => {
                assert_eq!(instrument, "Lumos");
                assert_eq!(energy, Some(28.0));
            }
            MsContext::Named(name) => panic!("read a factor list as the setup {name:?}"),
        }

        let named: MsContextArg = "Evosep60SPD_heron".parse().unwrap();
        match named.resolved {
            MsContext::Named(name) => assert_eq!(name, "Evosep60SPD_heron"),
            MsContext::Factors { .. } => panic!("read a setup name as factors"),
        }
    }

    #[test]
    fn a_half_written_factor_list_is_an_error_rather_than_a_setup_name() {
        // The failure this guards: silently treating "Lumos::FTMS::HCD" as a setup name would
        // report an unknown setup, sending the user looking for the wrong mistake.
        assert!("Lumos::FTMS::HCD".parse::<MsContextArg>().is_err());
        assert!("".parse::<MsContextArg>().is_err());
    }
}
