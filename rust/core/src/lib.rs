//! Pure-Rust pepdistill inference for the `predict` CLI.
//!
//! Load a `.safetensors` artifact ([`artifact::Artifact`]), run the transformer student forward
//! ([`model::Predictor`]), and assemble a single-peptide [`Prediction`] whose fragment table
//! matches `predict_library_fast`'s ordering and base-peak normalization.

pub mod artifact;
pub mod bucket;
pub mod chem;
pub mod composition;
pub mod model;
pub mod peptide;
pub mod tokenize;
pub mod unimod;

use anyhow::Result;

pub use artifact::Artifact;
pub use model::Predictor;

/// Struct-of-arrays fragment table (parallel columns), filtered to `rel >= min_intensity`.
pub struct Fragments {
    pub ion: Vec<String>,
    pub ord: Vec<i64>,
    pub z: Vec<i64>,
    pub mz: Vec<f64>,
    pub rel: Vec<f64>,
}

pub struct Prediction {
    pub peptide: String,
    pub charge: i64,
    pub precursor_mz: f64,
    pub rt: f32,
    pub ccs: f32,
    pub fragments: Fragments,
}

/// Optional MS acquisition context, parsed by the caller.
pub struct MsContext {
    pub instrument: String,
    pub detector: String,
    pub fragmentation: String,
    pub energy: Option<f32>,
}

/// Run one peptide end to end. `peptide` is a modified sequence in the form
/// [`peptide::Peptide::modified_sequence`] renders, e.g. `"[TMT6plex]PEPC[Carbamidomethyl@C]IDER"`.
pub fn predict(
    art: &Artifact,
    peptide: &str,
    charge: i64,
    ms_context: Option<&MsContext>,
    chrom_context: Option<&str>,
    min_intensity: f64,
) -> Result<Prediction> {
    // `peptide` is a modified sequence, not a bare one: parsing is what puts the mods on the
    // sites the runtime will encode. An unparseable string is an error, never a bare fallback.
    let pep = peptide::Peptide::parse(peptide)?;
    if pep.sequence.len() < 2 {
        anyhow::bail!("peptide must have at least 2 residues");
    }
    let rm = pep.residue_masses()?;

    let predictor = Predictor::new(art);
    let ms_vec = match ms_context {
        Some(c) => Some(predictor.encode_ms_context(
            &c.instrument,
            &c.detector,
            &c.fragmentation,
            c.energy,
        )?),
        None => None,
    };
    let chrom_vec = match chrom_context {
        Some(name) => Some(predictor.chrom_context(name)?),
        None => None,
    };

    let (ms2, rt, ccs) = predictor.forward(&pep, charge, ms_vec.as_ref(), chrom_vec.as_ref())?;

    let mz = chem::fragment_mz_matrix(&rm); // [L-1, n_ion]
    let frag_pos = pep.sequence.len() - 1;

    // Base peak over the whole spectrum (matches predict_library_fast).
    let peak = ms2.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let peak = if peak <= 0.0 { 1.0 } else { peak };

    let mut f = Fragments {
        ion: vec![],
        ord: vec![],
        z: vec![],
        mz: vec![],
        rel: vec![],
    };
    // Position-major, then ION_TYPES column order — same flatten as the Python reference.
    for i in 0..frag_pos {
        for (col, &(is_b, z)) in chem::ION_TYPES.iter().enumerate() {
            let rel = (ms2[[i, col]] / peak) as f64;
            if rel < min_intensity {
                continue;
            }
            let ordinal = if is_b {
                (i + 1) as i64
            } else {
                (frag_pos - i) as i64
            };
            f.ion.push(if is_b { "b" } else { "y" }.to_string());
            f.ord.push(ordinal);
            f.z.push(z as i64);
            f.mz.push(mz[i][col]);
            f.rel.push(rel);
        }
    }

    Ok(Prediction {
        // Report the re-rendered form, not the caller's string, so the output shows how the
        // input was actually read (e.g. a trailing `[mod]` resolved to the C-terminus).
        peptide: pep.modified_sequence(),
        charge,
        precursor_mz: chem::precursor_mz(&rm, charge),
        rt,
        ccs,
        fragments: f,
    })
}
