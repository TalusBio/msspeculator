//! pepdistill Rust inference CLI: FASTA -> DIA-NN TSV library, or one peptide -> JSON.

use anyhow::{anyhow, Result};
use clap::Parser;
use pepdistill_core::{predict, Artifact, MsContext, Prediction};
use serde_json::json;

mod library;

#[derive(Parser)]
#[command(
    name = "pepdistill-cli",
    about = "Generate a DIA-NN TSV library from FASTA or predict one peptide."
)]
struct Args {
    /// Path to the .safetensors artifact (from `pepdistill export-rust`).
    #[arg(long)]
    model: String,
    /// Peptide as a modified sequence: uppercase residues with optional `[mod]` brackets.
    ///
    /// A leading bracket is the N-terminus and a trailing one the C-terminus; a bracket after
    /// a residue modifies that residue. A body starting with `+`/`-` is a bare mass delta in
    /// Daltons, routed through the mass encoder instead of the compositional one. The
    /// `peptide` field of the JSON output echoes back how the string was read.
    ///
    /// Examples: `PEPTIDER`, `PEPC[Carbamidomethyl@C]IDER`, `[TMT6plex]PEPTIDER`,
    /// `PEP[+42.010565]TIDER`.
    #[arg(long)]
    peptide: Option<String>,
    /// Precursor charge.
    #[arg(long)]
    charge: Option<i64>,
    /// Digest this FASTA and write a DIA-NN TSV spectral library using Rust inference.
    #[arg(long)]
    fasta: Option<String>,
    /// Output path for --fasta library generation.
    #[arg(long)]
    out: Option<String>,
    /// Override the artifact activation for a controlled inference benchmark.
    #[arg(long, value_name = "ACTIVATION")]
    activation: Option<String>,
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
    /// Full acquisition context "INSTRUMENT::DETECTOR::FRAGMENTATION::ENERGY".
    #[arg(long)]
    ms_context: Option<String>,
    /// Shorthand: collision energy only (unknown instrument/detector/fragmentation).
    #[arg(long)]
    nce: Option<f32>,
    /// Named chromatography context (dataset) for raw RT; absent -> iRT base.
    #[arg(long)]
    chrom_context: Option<String>,
    /// Drop fragments below this base-peak-relative intensity.
    #[arg(long, default_value_t = 0.01)]
    min_intensity: f64,
}

fn parse_ms_context(args: &Args) -> Result<Option<MsContext>> {
    if let Some(spec) = &args.ms_context {
        let parts: Vec<&str> = spec.split("::").collect();
        if parts.len() != 4 {
            return Err(anyhow!(
                "--ms-context must be 'INSTRUMENT::DETECTOR::FRAGMENTATION::ENERGY'"
            ));
        }
        let energy: f32 = parts[3]
            .parse()
            .map_err(|_| anyhow!("energy {:?} is not a number", parts[3]))?;
        return Ok(Some(MsContext {
            instrument: parts[0].to_string(),
            detector: parts[1].to_string(),
            fragmentation: parts[2].to_string(),
            energy: Some(energy),
        }));
    }
    if let Some(nce) = args.nce {
        return Ok(Some(MsContext {
            instrument: String::new(),
            detector: String::new(),
            fragmentation: String::new(),
            energy: Some(nce),
        }));
    }
    Ok(None)
}

fn to_json(
    p: &Prediction,
    ms_context: Option<&String>,
    chrom: Option<&String>,
) -> serde_json::Value {
    json!({
        "peptide": p.peptide,
        "charge": p.charge,
        "precursor_mz": p.precursor_mz,
        "rt": p.rt,
        "ccs": p.ccs,
        "ms_context": ms_context,
        "chrom_context": chrom,
        "fragments": {
            "ion": p.fragments.ion,
            "ord": p.fragments.ord,
            "z": p.fragments.z,
            "mz": p.fragments.mz,
            "rel": p.fragments.rel,
        }
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let ms_ctx = parse_ms_context(&args)?;
    if let Some(fasta) = args.fasta.as_deref() {
        if args.peptide.is_some() || args.charge.is_some() {
            return Err(anyhow!(
                "--fasta cannot be combined with --peptide/--charge"
            ));
        }
        let out = args
            .out
            .as_deref()
            .ok_or_else(|| anyhow!("--fasta requires --out"))?;
        let stats = library::write_diann_tsv(&library::LibraryOptions {
            model: &args.model,
            fasta,
            out,
            activation: args.activation.as_deref(),
            ms_context: ms_ctx.as_ref(),
            chrom_context: args.chrom_context.as_deref(),
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
            stats.proteins, stats.peptides, stats.precursors, stats.fragments, out
        );
        return Ok(());
    }
    let peptide = args
        .peptide
        .as_deref()
        .ok_or_else(|| anyhow!("provide --peptide or --fasta"))?;
    let charge = args
        .charge
        .ok_or_else(|| anyhow!("--peptide requires --charge"))?;
    if args.out.is_some() {
        return Err(anyhow!("--out is only valid with --fasta"));
    }
    let mut art = Artifact::load(&args.model)?;
    library::apply_activation_override(&mut art, args.activation.as_deref())?;
    let pred = predict(
        &art,
        peptide,
        charge,
        ms_ctx.as_ref(),
        args.chrom_context.as_deref(),
        args.min_intensity,
    )?;
    let out = to_json(&pred, args.ms_context.as_ref(), args.chrom_context.as_ref());
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
