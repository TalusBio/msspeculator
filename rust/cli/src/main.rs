//! pepdistill Rust inference CLI: FASTA libraries, peptide prediction, and model diagnostics.

use std::str::FromStr;

use anyhow::Result;
use clap::{Parser, Subcommand};
use pepdistill_core::{predict, Artifact, MsContext, Prediction};
use serde_json::json;

mod diagnostics;
mod library;

#[derive(Parser)]
#[command(
    name = "pepdistill-cli",
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
}

#[derive(clap::Args)]
struct ArtifactArgs {
    /// Path to the .safetensors artifact (from `pepdistill export-rust`).
    #[arg(long)]
    model: String,
    /// Override the artifact activation for a controlled inference benchmark.
    #[arg(long, value_name = "ACTIVATION")]
    activation: Option<String>,
}

#[derive(Clone)]
struct FullMsContext {
    raw: String,
    instrument: String,
    detector: String,
    fragmentation: String,
    energy: f32,
}

impl FromStr for FullMsContext {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = value.split("::").collect();
        if parts.len() != 4 {
            return Err("expected INSTRUMENT::DETECTOR::FRAGMENTATION::ENERGY".to_string());
        }
        let energy = parts[3]
            .parse()
            .map_err(|_| format!("energy {:?} is not a number", parts[3]))?;
        Ok(Self {
            raw: value.to_string(),
            instrument: parts[0].to_string(),
            detector: parts[1].to_string(),
            fragmentation: parts[2].to_string(),
            energy,
        })
    }
}

#[derive(clap::Args)]
struct ContextArgs {
    /// Full acquisition context "INSTRUMENT::DETECTOR::FRAGMENTATION::ENERGY".
    #[arg(long, conflicts_with = "nce")]
    ms_context: Option<FullMsContext>,
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
            .map(|context| MsContext {
                instrument: context.instrument.clone(),
                detector: context.detector.clone(),
                fragmentation: context.fragmentation.clone(),
                energy: Some(context.energy),
            })
            .or_else(|| {
                self.nce.map(|energy| MsContext {
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
    /// Peptide as a modified sequence, for example `PEPC[Carbamidomethyl@C]IDER`.
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
    fasta: String,
    /// DIA-NN TSV output path.
    #[arg(long)]
    out: String,
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
    /// Maximum number of variable Oxidation@M modifications per peptide.
    #[arg(long, default_value_t = 1)]
    max_variable_oxidation: usize,
    /// Do not apply fixed Carbamidomethyl@C during FASTA library generation.
    #[arg(long)]
    no_fixed_carbamidomethyl: bool,
    #[command(flatten)]
    context: ContextArgs,
    /// Drop fragments below this base-peak-relative intensity.
    #[arg(long, default_value_t = 0.01)]
    min_intensity: f64,
}

#[derive(clap::Args)]
struct DoctorArgs {
    #[command(flatten)]
    artifact: ArtifactArgs,
    /// Directory for diagnostic artifacts.
    #[arg(long, default_value = "model-doctor")]
    out: String,
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
    let mut artifact = Artifact::load(&args.artifact.model)?;
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

fn run_library(args: LibraryArgs) -> Result<()> {
    let ms_context = args.context.ms_context();
    let stats = library::write_diann_tsv(&library::LibraryOptions {
        model: &args.artifact.model,
        fasta: &args.fasta,
        out: &args.out,
        activation: args.artifact.activation.as_deref(),
        ms_context: ms_context.as_ref(),
        chrom_context: args.context.chrom_context.as_deref(),
        min_intensity: args.min_intensity,
        missed_cleavages: args.missed_cleavages,
        min_length: args.min_length,
        max_length: args.max_length,
        min_charge: args.min_charge,
        max_charge: args.max_charge,
        max_variable_oxidation: args.max_variable_oxidation,
        no_fixed_carbamidomethyl: args.no_fixed_carbamidomethyl,
    })?;
    eprintln!(
        "{} proteins -> {} peptides -> {} precursors -> {} fragments -> {}",
        stats.proteins, stats.peptides, stats.precursors, stats.fragments, args.out
    );
    Ok(())
}

fn run_doctor(args: DoctorArgs) -> Result<()> {
    let mut artifact = Artifact::load(&args.artifact.model)?;
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
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Predict(args) => run_predict(args),
        Command::Library(args) => run_library(args),
        Command::RunDoctor(args) => run_doctor(args),
    }
}
