//! Pure-Rust pepdistill inference for the `predict` CLI.
//!
//! Load a `.safetensors` artifact ([`artifact::Artifact`]), run the transformer student forward
//! ([`model::Predictor`]), and assemble a single-peptide [`Prediction`] whose fragment table
//! matches `predict_library_fast`'s ordering and base-peak normalization.

pub mod artifact;
pub mod bucket;
pub mod builtin;
pub mod chem;
pub mod composition;
pub mod fit;
pub mod landmarks;
pub mod model;
pub mod peptide;
pub mod proforma;
pub mod speclib;
pub mod split;
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
    /// Retention as the requested context reports it: a dataset's own gradient time in minutes
    /// when a chromatography context was named, otherwise the context-free index.
    pub rt: f32,
    /// The context-free index, carried only when `rt` is *not* it -- that is, when a
    /// chromatography context was applied. A consumer then has both the duration and the index
    /// without having to run the model twice or know which one `rt` holds.
    pub irt: Option<f32>,
    pub ccs: f32,
    pub fragments: Fragments,
}

/// How the caller addressed the MS acquisition side.
///
/// Two ways in, not two spellings of one: `Factors` composes the context out of recorded
/// metadata through the encoder, which is available only because the source recorded that
/// metadata. `Named` is for a source that recorded none — a published library reports no
/// instrument and no collision energy — and points at a row fitted against its spectra.
#[derive(Clone)]
pub enum MsContext {
    Factors {
        instrument: String,
        detector: String,
        fragmentation: String,
        energy: Option<f32>,
    },
    Named(String),
}

/// Context terms resolved once for repeated predictions with one acquisition setup.
#[derive(Clone)]
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
        // Both arms end at the same projection: a fitted row lives in the space the encoder
        // produces, so it has to reach the heads the way an encoded context does.
        let ms_shift = match ms_context {
            Some(MsContext::Factors {
                instrument,
                detector,
                fragmentation,
                energy,
            }) => Some(predictor.ms_context_shift(instrument, detector, fragmentation, *energy)?),
            Some(MsContext::Named(setup)) => {
                Some(predictor.context_shift(predictor.named_ms_context(setup)?.view())?)
            }
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

    /// Whether retention will come out as a dataset's gradient time rather than as the index.
    fn shifts_retention(&self) -> bool {
        self.chrom_shift.is_some() || self.chrom_affine.is_some()
    }
}

/// Run one peptide end to end. `peptide` is a modified sequence in the form
/// [`peptide::Peptide::modified_sequence`] renders, e.g.
/// `"[UNIMOD:737]-PEPC[UNIMOD:4]IDER"`.
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
    Ok(predictions
        .pop()
        .expect("one requested charge yields one prediction"))
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
    let rt = predictor.predict_rt(&encoded, context.chrom_shift.as_ref(), context.chrom_affine)?;
    // The chromatography context enters through the head's own input, not as an affine on its
    // output, so the index cannot be recovered from `rt` -- it takes a second pass. The RT head
    // is one small projection on an already-encoded peptide, unlike the per-position MS2 head.
    let irt = if context.shifts_retention() {
        Some(predictor.predict_rt(&encoded, None, None)?)
    } else {
        None
    };
    let mz = chem::fragment_mz_matrix(&rm); // [L-1, n_ion]
    let frag_pos = pep.sequence.len() - 1;
    let modified_sequence = pep.modified_sequence();
    let mut predictions = Vec::with_capacity(charges.len());

    let charge_outputs = predictor.predict_charges(&encoded, charges, context.ms_shift.as_ref())?;
    for (&charge, (ms2, ccs)) in charges.iter().zip(charge_outputs) {
        predictions.push(assemble_prediction(
            &modified_sequence,
            frag_pos,
            &rm,
            &mz,
            charge,
            ms2,
            rt,
            irt,
            ccs,
            min_intensity,
        ));
    }

    Ok(predictions)
}

/// Predict a same-length peptide batch while sharing every dense transformer/head projection.
pub fn predict_peptide_batch_charges_prepared(
    art: &Artifact,
    peptides: &[peptide::Peptide],
    charges: &[i64],
    context: &PreparedContext,
    min_intensity: f64,
) -> Result<Vec<Vec<Prediction>>> {
    if peptides.is_empty() {
        return Ok(Vec::new());
    }
    let seq_len = peptides[0].sequence.len();
    if seq_len < 2 || peptides.iter().any(|pep| pep.sequence.len() != seq_len) {
        anyhow::bail!("batch prediction requires peptides of one shared length >= 2");
    }
    let residue_masses = peptides
        .iter()
        .map(peptide::Peptide::residue_masses)
        .collect::<Result<Vec<_>>>()?;
    let fragment_mz = residue_masses
        .iter()
        .map(|rm| chem::fragment_mz_matrix(rm))
        .collect::<Vec<_>>();
    let modified_sequences = peptides
        .iter()
        .map(peptide::Peptide::modified_sequence)
        .collect::<Vec<_>>();

    let predictor = Predictor::new(art);
    let encoded = predictor.encode_batch(peptides)?;
    let rt =
        predictor.predict_rt_batch(&encoded, context.chrom_shift.as_ref(), context.chrom_affine)?;
    // See the single-peptide path: one extra pass through the RT head, shared across the batch.
    let irt = if context.shifts_retention() {
        Some(predictor.predict_rt_batch(&encoded, None, None)?)
    } else {
        None
    };
    let charge_outputs =
        predictor.predict_batch_charges(&encoded, charges, context.ms_shift.as_ref())?;
    let frag_pos = seq_len - 1;
    let mut result = Vec::with_capacity(peptides.len());
    for peptide_i in 0..peptides.len() {
        let mut predictions = Vec::with_capacity(charges.len());
        for (&charge, (ms2, ccs)) in charges.iter().zip(charge_outputs[peptide_i].iter()) {
            predictions.push(assemble_prediction(
                &modified_sequences[peptide_i],
                frag_pos,
                &residue_masses[peptide_i],
                &fragment_mz[peptide_i],
                charge,
                ms2.clone(),
                rt[peptide_i],
                irt.as_ref().map(|irt| irt[peptide_i]),
                *ccs,
                min_intensity,
            ));
        }
        result.push(predictions);
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn assemble_prediction(
    modified_sequence: &str,
    frag_pos: usize,
    residue_masses: &[f64],
    fragment_mz: &[Vec<f64>],
    charge: i64,
    ms2: ndarray::Array2<f32>,
    rt: f32,
    irt: Option<f32>,
    ccs: f32,
    min_intensity: f64,
) -> Prediction {
    let peak = ms2.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let peak = if peak <= 0.0 { 1.0 } else { peak };
    let mut f = Fragments {
        ion: vec![],
        ord: vec![],
        z: vec![],
        mz: vec![],
        rel: vec![],
    };
    for i in 0..frag_pos {
        for (col, &(is_b, z)) in chem::ION_TYPES.iter().enumerate() {
            let rel = (ms2[[i, col]] / peak) as f64;
            if rel < min_intensity {
                continue;
            }
            f.ion.push(if is_b { "b" } else { "y" }.to_string());
            f.ord.push(if is_b {
                (i + 1) as i64
            } else {
                (frag_pos - i) as i64
            });
            f.z.push(z as i64);
            f.mz.push(fragment_mz[i][col]);
            f.rel.push(rel);
        }
    }
    Prediction {
        peptide: modified_sequence.to_string(),
        charge,
        precursor_mz: chem::precursor_mz(residue_masses, charge),
        rt,
        irt,
        ccs,
        fragments: f,
    }
}
