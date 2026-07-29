//! pepdistill Rust predict CLI: one peptide -> JSON prediction on stdout.

use anyhow::{anyhow, Result};
use clap::Parser;
use pepdistill_core::{predict, Artifact, MsContext, Prediction};
use serde_json::json;

#[derive(Parser)]
#[command(
    name = "pepdistill-cli",
    about = "Predict MS2/RT/CCS for one peptide (torch parity)."
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
    peptide: String,
    /// Precursor charge.
    #[arg(long)]
    charge: i64,
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
    let art = Artifact::load(&args.model)?;
    let ms_ctx = parse_ms_context(&args)?;
    let pred = predict(
        &art,
        &args.peptide,
        args.charge,
        ms_ctx.as_ref(),
        args.chrom_context.as_deref(),
        args.min_intensity,
    )?;
    let out = to_json(&pred, args.ms_context.as_ref(), args.chrom_context.as_ref());
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
