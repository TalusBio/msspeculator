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
use ndarray::Array1;

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

/// Context terms resolved once for repeated predictions with one acquisition setup.
pub struct PreparedContext {
    ms_shift: Option<Array1<f32>>,
    chrom_shift: Option<Array1<f32>>,
    chrom_affine: Option<(f32, f32)>,
}

impl PreparedContext {
    pub fn new(
        art: &Artifact,
        ms_context: Option<&MsContext>,
        chrom_context: Option<&str>,
    ) -> Result<Self> {
        let predictor = Predictor::new(art);
        let ms_shift = match ms_context {
            Some(c) => Some(predictor.ms_context_shift(
                &c.instrument,
                &c.detector,
                &c.fragmentation,
                c.energy,
            )?),
            None => None,
        };
        // A named dataset supplies both its feature shift and output affine. Resolving only one
        // would silently apply half of the chromatography adaptation fitted during training.
        let (chrom_shift, chrom_affine) = match chrom_context {
            Some(name) => (
                Some(predictor.chrom_context_shift(name)?),
                Some(predictor.chrom_affine(name)?),
            ),
            None => (None, None),
        };
        Ok(Self {
            ms_shift,
            chrom_shift,
            chrom_affine,
        })
    }
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
    predict_peptide(art, &pep, charge, ms_context, chrom_context, min_intensity)
}

/// Run an already-constructed peptide end to end. This avoids round-tripping through the
/// modified-sequence grammar, which intentionally renders a final-residue modification in the
/// same position as a C-terminal modification and therefore cannot disambiguate those cases.
pub fn predict_peptide(
    art: &Artifact,
    pep: &peptide::Peptide,
    charge: i64,
    ms_context: Option<&MsContext>,
    chrom_context: Option<&str>,
    min_intensity: f64,
) -> Result<Prediction> {
    let context = PreparedContext::new(art, ms_context, chrom_context)?;
    predict_peptide_prepared(art, pep, charge, &context, min_intensity)
}

/// Run a peptide using context that has already been resolved for a bulk prediction job.
pub fn predict_peptide_prepared(
    art: &Artifact,
    pep: &peptide::Peptide,
    charge: i64,
    context: &PreparedContext,
    min_intensity: f64,
) -> Result<Prediction> {
    let mut predictions =
        predict_peptide_charges_prepared(art, pep, &[charge], context, min_intensity)?;
    Ok(predictions.pop().expect("one requested charge yields one prediction"))
}

/// Encode a peptide once and run its charge-dependent heads for every requested charge.
pub fn predict_peptide_charges_prepared(
    art: &Artifact,
    pep: &peptide::Peptide,
    charges: &[i64],
    context: &PreparedContext,
    min_intensity: f64,
) -> Result<Vec<Prediction>> {
    if pep.sequence.len() < 2 {
        anyhow::bail!("peptide must have at least 2 residues");
    }
    if charges.is_empty() {
        return Ok(Vec::new());
    }
    let rm = pep.residue_masses()?;

    let predictor = Predictor::new(art);
    let encoded = predictor.encode(pep)?;
    let rt = predictor.predict_rt(
        &encoded,
        context.chrom_shift.as_ref(),
        context.chrom_affine,
    )?;
    let mz = chem::fragment_mz_matrix(&rm); // [L-1, n_ion]
    let frag_pos = pep.sequence.len() - 1;
    let modified_sequence = pep.modified_sequence();
    let mut predictions = Vec::with_capacity(charges.len());

    let charge_outputs =
        predictor.predict_charges(&encoded, charges, context.ms_shift.as_ref())?;
    for (&charge, (ms2, ccs)) in charges.iter().zip(charge_outputs) {

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

        predictions.push(Prediction {
            // Report the re-rendered form, not the caller's string, so output shows how the
            // input was actually read (e.g. a trailing `[mod]` resolved to the C-terminus).
            peptide: modified_sequence.clone(),
            charge,
            precursor_mz: chem::precursor_mz(&rm, charge),
            rt,
            ccs,
            fragments: f,
        });
    }

    Ok(predictions)
}
