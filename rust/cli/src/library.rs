use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use anyhow::{bail, Context, Result};
use pepdistill_core::peptide::{ModSpec, Peptide, Site};
use pepdistill_core::proforma::{parse_modification_rule, ModificationRule, ModificationTarget};
use pepdistill_core::{
    predict_peptide_batch_charges_prepared, Artifact, MsContext, Prediction, PreparedContext,
};

const VALID_AA: &str = "GASPVTCLINDQKEMHFRYW";
const CCS_IM_COEF: f64 = 1059.62245;
const IM_GAS_MASS: f64 = 28.0;
const INFERENCE_BATCH_SIZE: usize = 64;

pub struct LibraryOptions<'a> {
    pub model: &'a str,
    pub fasta: &'a str,
    pub out: &'a str,
    pub activation: Option<&'a str>,
    pub ms_context: Option<&'a MsContext>,
    pub chrom_context: Option<&'a str>,
    pub min_intensity: f64,
    pub missed_cleavages: usize,
    pub min_length: usize,
    pub max_length: usize,
    pub min_charge: i64,
    pub max_charge: i64,
    pub fixed_mods: &'a [String],
    pub variable_mods: &'a [String],
    pub max_variable_mods: usize,
    /// Emit at most this many of the strongest fragments per precursor, or all of them when
    /// `None`. Applied after `min_intensity`, so both bound the transition list independently.
    pub max_fragments: Option<usize>,
    /// Where to write the resolved-configuration sidecar, or `None` to skip it.
    pub config_out: Option<&'a str>,
}

/// Apply a runtime activation override used for controlled inference benchmarks.
/// The artifact remains unchanged on disk; only this loaded instance is modified.
pub fn apply_activation_override(artifact: &mut Artifact, activation: Option<&str>) -> Result<()> {
    if let Some(activation) = activation {
        match activation {
            "gelu" | "gelu_tanh" | "relu" | "leaky_relu" => {}
            other => bail!(
                "unsupported activation {:?}; expected gelu, gelu_tanh, relu, or leaky_relu",
                other
            ),
        }
        artifact.meta.config.activation = activation.to_string();
    }
    Ok(())
}

/// Which serialization the library is written in.
///
/// Chosen from the output path rather than a flag of its own. The path already decides
/// compression (`.gz`), and a format flag that can disagree with the extension is a defect
/// waiting for a caller: `library.mzspeclib.txt` holding DIA-NN TSV is worse than no option.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LibraryFormat {
    DiannTsv,
    MzSpecLib,
}

impl LibraryFormat {
    pub fn for_output(out: &str) -> Self {
        let stem = out.strip_suffix(".gz").unwrap_or(out);
        if stem.ends_with(".mzspeclib.txt") || stem.ends_with(".mzspeclib") {
            Self::MzSpecLib
        } else {
            Self::DiannTsv
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::DiannTsv => "diann-tsv",
            Self::MzSpecLib => "mzspeclib-text",
        }
    }
}

/// One kept transition, borrowed from the prediction it came out of.
pub struct Peak<'a> {
    pub mz: f64,
    pub intensity: f64,
    pub ion: &'a str,
    pub ordinal: i64,
    pub charge: i64,
}

/// One validated precursor, ready for any serialization.
///
/// Every field is already checked to be physical and the transition list is already capped, so a
/// sink's only job is spelling: no sink can disagree with another about what the library contains.
pub struct SpectrumRow<'a> {
    pub stripped: &'a str,
    pub protein_group: &'a str,
    pub diann_sequence: &'a str,
    /// ProForma spelling of the same peptidoform, e.g. `PEPC[UNIMOD:4]IDER`.
    pub proforma: &'a str,
    pub charge: i64,
    pub precursor_mz: f64,
    /// Monoisotopic mass of the neutral peptidoform, which is what a library states about an
    /// analyte; the m/z above is the charged species this entry was predicted for.
    pub neutral_mass: f64,
    /// Retention as the requested context reports it: a dataset's gradient time in minutes with
    /// `--chrom-context`, otherwise the context-free index.
    pub rt: f32,
    /// The context-free index, present only when `rt` is a gradient time instead. Its presence is
    /// what tells a sink which quantity it holds, so there is no second flag to disagree with.
    pub irt: Option<f32>,
    pub mobility: f64,
    pub peaks: Vec<Peak<'a>>,
}

/// A library serialization: one header, then one entry per precursor, in prediction order.
pub trait LibrarySink {
    /// `config` is the resolved configuration from [`resolve_config`], for formats that can carry
    /// their own provenance. A format with nowhere to put it ignores the argument.
    fn header(&mut self, config: &serde_json::Value) -> Result<()>;
    fn spectrum(&mut self, row: &SpectrumRow<'_>) -> Result<()>;
    /// Flush the stream. Called once, after the last spectrum: a buffered library that is never
    /// flushed is a truncated library.
    fn finish(&mut self) -> Result<()>;
}

struct DiannSink<W: Write> {
    writer: W,
}

impl<W: Write> LibrarySink for DiannSink<W> {
    fn header(&mut self, _config: &serde_json::Value) -> Result<()> {
        writeln!(self.writer, "ModifiedPeptide\tStrippedPeptide\tPrecursorMz\tPrecursorCharge\tTr_recalibrated\tIonMobility\tProteinID\tDecoy\tFragmentMz\tFragmentType\tFragmentNumber\tFragmentCharge\tFragmentLossType\tRelativeIntensity")?;
        Ok(())
    }

    fn spectrum(&mut self, row: &SpectrumRow<'_>) -> Result<()> {
        for peak in &row.peaks {
            writeln!(
                self.writer,
                "{}\t{}\t{:.8}\t{}\t{:.6}\t{:.8}\t{}\t0\t{:.8}\t{}\t{}\t{}\tnoloss\t{:.8}",
                row.diann_sequence,
                row.stripped,
                row.precursor_mz,
                row.charge,
                row.rt,
                row.mobility,
                row.protein_group,
                peak.mz,
                peak.ion,
                peak.ordinal,
                peak.charge,
                peak.intensity,
            )?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct LibraryStats {
    pub proteins: usize,
    pub peptides: usize,
    pub precursors: usize,
    pub fragments: usize,
}

fn parse_fasta(path: &Path) -> Result<Vec<(String, String)>> {
    let reader = BufReader::new(
        File::open(path).with_context(|| format!("opening FASTA {}", path.display()))?,
    );
    let mut records = Vec::new();
    let mut id: Option<String> = None;
    let mut seq = String::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("reading {}:{}", path.display(), line_no + 1))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(header) = line.strip_prefix('>') {
            if let Some(old_id) = id.replace(
                header
                    .split_whitespace()
                    .next()
                    .context("empty FASTA header")?
                    .to_string(),
            ) {
                records.push((old_id, std::mem::take(&mut seq)));
            }
        } else {
            if id.is_none() {
                bail!(
                    "{}:{} has sequence before the first FASTA header",
                    path.display(),
                    line_no + 1
                );
            }
            seq.push_str(&line.to_ascii_uppercase());
        }
    }
    if let Some(id) = id {
        records.push((id, seq));
    }
    if records.is_empty() {
        bail!("FASTA {} contains no records", path.display());
    }
    Ok(records)
}

fn digest_tryptic(sequence: &str, missed: usize, min_len: usize, max_len: usize) -> Vec<String> {
    let bytes = sequence.as_bytes();
    let mut sites = vec![0usize];
    for (i, &aa) in bytes.iter().enumerate() {
        if (aa == b'K' || aa == b'R') && bytes.get(i + 1) != Some(&b'P') {
            sites.push(i + 1);
        }
    }
    if sites.last().copied() != Some(bytes.len()) {
        sites.push(bytes.len());
    }
    let mut out = Vec::new();
    for start in 0..sites.len().saturating_sub(1) {
        for mc in 0..=missed {
            let end = start + mc + 1;
            if end >= sites.len() {
                break;
            }
            let pep = &sequence[sites[start]..sites[end]];
            if (min_len..=max_len).contains(&pep.len())
                && pep.bytes().all(|aa| VALID_AA.as_bytes().contains(&aa))
            {
                out.push(pep.to_string());
            }
        }
    }
    out
}

fn peptide_proteins(
    records: &[(String, String)],
    missed: usize,
    min_len: usize,
    max_len: usize,
) -> BTreeMap<String, BTreeSet<String>> {
    let mut peptides: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (protein, sequence) in records {
        for peptide in digest_tryptic(sequence, missed, min_len, max_len) {
            peptides.entry(peptide).or_default().insert(protein.clone());
        }
    }
    peptides
}

fn rule_sites(sequence: &str, rule: &ModificationRule) -> Vec<Site> {
    match &rule.target {
        ModificationTarget::Residues(targets) => sequence
            .chars()
            .enumerate()
            .filter_map(|(i, aa)| targets.contains(&aa).then_some(Site::Residue(i)))
            .collect(),
        ModificationTarget::PeptideNTerm => vec![Site::NTerm],
        ModificationTarget::PeptideCTerm => vec![Site::CTerm],
    }
}

fn target_overlap(a: &ModificationTarget, b: &ModificationTarget) -> bool {
    match (a, b) {
        (ModificationTarget::Residues(a), ModificationTarget::Residues(b)) => !a.is_disjoint(b),
        (ModificationTarget::PeptideNTerm, ModificationTarget::PeptideNTerm)
        | (ModificationTarget::PeptideCTerm, ModificationTarget::PeptideCTerm) => true,
        _ => false,
    }
}

fn annotation(spec: &ModSpec) -> String {
    match spec {
        ModSpec::Unimod { accession, .. } => format!("(UniMod:{accession})"),
        ModSpec::MassOnly(mass) => format!("({mass:+})"),
        ModSpec::Formula { formula, .. } => format!("[Formula:{formula}]"),
    }
}

fn render_diann(sequence: &str, mods: &[(Site, ModSpec)]) -> String {
    let mut diann = String::new();
    for (site, spec) in mods {
        if *site == Site::NTerm {
            diann.push_str(&annotation(spec));
        }
    }
    for (i, aa) in sequence.chars().enumerate() {
        diann.push(aa);
        for (site, spec) in mods {
            if *site == Site::Residue(i) {
                diann.push_str(&annotation(spec));
            }
        }
    }
    for (site, spec) in mods {
        if *site == Site::CTerm {
            diann.push_str(&annotation(spec));
        }
    }
    diann
}

fn modified_forms(
    sequence: &str,
    fixed_rules: &[ModificationRule],
    variable_rules: &[ModificationRule],
    max_variable: usize,
) -> Result<Vec<(Peptide, String)>> {
    let mut fixed = Vec::new();
    for rule in fixed_rules {
        fixed.extend(
            rule_sites(sequence, rule)
                .into_iter()
                .map(|site| (site, rule.spec.clone())),
        );
    }
    let candidates: Vec<(Site, ModSpec)> = variable_rules
        .iter()
        .flat_map(|rule| {
            rule_sites(sequence, rule)
                .into_iter()
                .map(|site| (site, rule.spec.clone()))
        })
        .collect();

    fn choose(
        candidates: &[(Site, ModSpec)],
        start: usize,
        left: usize,
        picked: &mut Vec<(Site, ModSpec)>,
        out: &mut Vec<Vec<(Site, ModSpec)>>,
    ) {
        if left == 0 {
            out.push(picked.clone());
            return;
        }
        for i in start..=candidates.len().saturating_sub(left) {
            if picked.iter().any(|(site, _)| *site == candidates[i].0) {
                continue;
            }
            picked.push(candidates[i].clone());
            choose(candidates, i + 1, left - 1, picked, out);
            picked.pop();
        }
    }

    let mut variable_forms = vec![Vec::new()];
    for n in 1..=max_variable.min(candidates.len()) {
        choose(&candidates, 0, n, &mut Vec::new(), &mut variable_forms);
    }
    variable_forms
        .into_iter()
        .map(|variable| {
            let peptide = Peptide::new(
                sequence.to_string(),
                fixed.iter().cloned().chain(variable).collect(),
            );
            peptide.validate_mod_specs()?;
            let diann = render_diann(sequence, &peptide.mods);
            Ok((peptide, diann))
        })
        .collect()
}

fn ccs_to_bruker_mobility(ccs: f64, charge: i64, precursor_mz: f64) -> f64 {
    let mass = precursor_mz * charge as f64;
    let reduced_mass = mass * IM_GAS_MASS / (mass + IM_GAS_MASS);
    ccs * reduced_mass.sqrt() / charge as f64 / CCS_IM_COEF
}

struct PendingPeptide {
    stripped: String,
    protein_group: String,
    peptide: Peptide,
    diann_sequence: String,
}

struct PredictedPeptide {
    stripped: String,
    protein_group: String,
    diann_sequence: String,
    predictions: Vec<Prediction>,
}

/// Validate one prediction and cap its transition list.
fn spectrum_row<'a>(
    item: &'a PredictedPeptide,
    prediction: &'a Prediction,
    max_fragments: Option<usize>,
) -> Result<SpectrumRow<'a>> {
    let diann_sequence = item.diann_sequence.as_str();
    let charge = prediction.charge;
    if !prediction.precursor_mz.is_finite()
        || !prediction.rt.is_finite()
        || !prediction.ccs.is_finite()
        || prediction.precursor_mz <= 0.0
        || prediction.ccs <= 0.0
    {
        bail!(
            "non-physical precursor prediction for {} charge {}: mz={}, rt={}, ccs={}",
            diann_sequence,
            charge,
            prediction.precursor_mz,
            prediction.rt,
            prediction.ccs
        );
    }
    let mobility = ccs_to_bruker_mobility(prediction.ccs as f64, charge, prediction.precursor_mz);
    if !mobility.is_finite() || mobility <= 0.0 {
        bail!(
            "non-physical mobility for {} charge {}: {}",
            diann_sequence,
            charge,
            mobility
        );
    }
    // Rank by intensity and keep the strongest `max_fragments`. Validation still runs over every
    // fragment below, including discarded ones: a non-physical m/z that happens to rank 16th is a
    // model or chemistry fault either way, and letting the cap hide it would make the error
    // depend on how many transitions the caller asked for.
    let mut keep = vec![true; prediction.fragments.mz.len()];
    if let Some(limit) = max_fragments {
        if keep.len() > limit {
            let mut order: Vec<usize> = (0..keep.len()).collect();
            order.sort_by(|a, b| {
                prediction.fragments.rel[*b]
                    .partial_cmp(&prediction.fragments.rel[*a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    // Ties broken by index so a regenerated shard keeps the same peaks.
                    .then(a.cmp(b))
            });
            keep = vec![false; keep.len()];
            for index in &order[..limit] {
                keep[*index] = true;
            }
        }
    }
    let mut peaks = Vec::new();
    #[allow(clippy::needless_range_loop)] // `i` indexes four parallel fragment columns, not one
    for i in 0..prediction.fragments.mz.len() {
        let fragment_mz = prediction.fragments.mz[i];
        let intensity = prediction.fragments.rel[i];
        if !fragment_mz.is_finite()
            || fragment_mz <= 0.0
            || !intensity.is_finite()
            || !(0.0..=1.0).contains(&intensity)
        {
            bail!(
                "invalid fragment for {} charge {} at index {}: mz={}, intensity={}",
                diann_sequence,
                charge,
                i,
                fragment_mz,
                intensity
            );
        }
        // Validated above, but outside the strongest `max_fragments`.
        if !keep[i] {
            continue;
        }
        peaks.push(Peak {
            mz: fragment_mz,
            intensity,
            ion: prediction.fragments.ion[i].as_str(),
            ordinal: prediction.fragments.ord[i],
            charge: prediction.fragments.z[i],
        });
    }
    Ok(SpectrumRow {
        stripped: item.stripped.as_str(),
        protein_group: item.protein_group.as_str(),
        diann_sequence,
        proforma: prediction.peptide.as_str(),
        charge,
        precursor_mz: prediction.precursor_mz,
        neutral_mass: (prediction.precursor_mz - pepdistill_core::chem::PROTON) * charge as f64,
        rt: prediction.rt,
        irt: prediction.irt,
        mobility,
        peaks,
    })
}

fn predict_batch(
    artifact: &Artifact,
    context: &PreparedContext,
    charges: &[i64],
    min_intensity: f64,
    batch: Vec<PendingPeptide>,
) -> Result<Vec<PredictedPeptide>> {
    let (metadata, peptides): (Vec<_>, Vec<_>) = batch
        .into_iter()
        .map(|item| {
            (
                (item.stripped, item.protein_group, item.diann_sequence),
                item.peptide,
            )
        })
        .unzip();
    let predictions = predict_peptide_batch_charges_prepared(
        artifact,
        &peptides,
        charges,
        context,
        min_intensity,
    )?;
    Ok(metadata
        .into_iter()
        .zip(predictions)
        .map(
            |((stripped, protein_group, diann_sequence), predictions)| PredictedPeptide {
                stripped,
                protein_group,
                diann_sequence,
                predictions,
            },
        )
        .collect())
}

fn write_predicted_batch(
    sink: &mut dyn LibrarySink,
    stats: &mut LibraryStats,
    batch: &[PredictedPeptide],
    max_fragments: Option<usize>,
) -> Result<()> {
    for item in batch {
        for prediction in &item.predictions {
            let row = spectrum_row(item, prediction, max_fragments)?;
            stats.precursors += 1;
            stats.fragments += row.peaks.len();
            sink.spectrum(&row)?;
        }
    }
    Ok(())
}

pub fn write_library(opts: &LibraryOptions<'_>) -> Result<LibraryStats> {
    if !(0.0..=1.0).contains(&opts.min_intensity) {
        bail!(
            "min_intensity must be in [0, 1], got {}",
            opts.min_intensity
        );
    }
    if opts.min_charge < 1 || opts.max_charge < opts.min_charge {
        bail!(
            "invalid charge range {}..={}",
            opts.min_charge,
            opts.max_charge
        );
    }
    if opts.min_length < 2 || opts.max_length < opts.min_length {
        bail!(
            "invalid peptide length range {}..={}",
            opts.min_length,
            opts.max_length
        );
    }
    let fixed_rules = opts
        .fixed_mods
        .iter()
        .map(|rule| parse_modification_rule(rule))
        .collect::<Result<Vec<_>>>()?;
    let variable_rules = opts
        .variable_mods
        .iter()
        .map(|rule| parse_modification_rule(rule))
        .collect::<Result<Vec<_>>>()?;
    for fixed in &fixed_rules {
        for variable in &variable_rules {
            if target_overlap(&fixed.target, &variable.target) {
                bail!(
                    "fixed and variable modification rules overlap ({:?} and {:?}); \
                     use --no-fixed-mods or choose disjoint targets",
                    fixed.target,
                    variable.target
                );
            }
        }
    }
    let records = parse_fasta(Path::new(opts.fasta))?;
    let peptides = peptide_proteins(
        &records,
        opts.missed_cleavages,
        opts.min_length,
        opts.max_length,
    );
    if peptides.is_empty() {
        bail!("FASTA digest produced no peptides");
    }
    let model = pepdistill_core::builtin::load_model(opts.model)?;
    let mut artifact = model.artifact;
    apply_activation_override(&mut artifact, opts.activation)?;
    let context = PreparedContext::new(&artifact, opts.ms_context, opts.chrom_context)?;
    let charges = (opts.min_charge..=opts.max_charge).collect::<Vec<_>>();
    let stats = LibraryStats {
        proteins: records.len(),
        peptides: peptides.len(),
        ..LibraryStats::default()
    };

    let worker_count = std::env::var("PEPDISTILL_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .or_else(|| {
            thread::available_parallelism()
                .ok()
                .map(|n| n.get().saturating_sub(1).max(1))
        })
        .unwrap_or(1);
    let queue_capacity = worker_count * 2;
    let (work_tx, work_rx) = mpsc::sync_channel::<Option<Vec<PendingPeptide>>>(queue_capacity);
    let work_rx = Arc::new(Mutex::new(work_rx));
    let (result_tx, result_rx) =
        mpsc::sync_channel::<Result<Vec<PredictedPeptide>>>(queue_capacity);
    let artifact = Arc::new(artifact);

    // Copied out of `opts` before the threads start: the writer closure is `move`, so borrowing
    // `opts` there would outlive it.
    let max_fragments = opts.max_fragments;
    let format = LibraryFormat::for_output(opts.out);
    let out_path = opts.out.to_string();
    // Resolved before the first spectrum because an mzSpecLib header carries it, and a header is
    // the first thing on the stream. The sidecar reuses the same value, so the copy bundled in
    // the library and the copy beside it are the same copy.
    let config = resolve_config(opts, format, &artifact, &model.digest)?;
    let header_config = config.clone();
    let writer_file =
        File::create(opts.out).with_context(|| format!("creating library {}", opts.out))?;
    // A `.gz` suffix compresses in the writer thread rather than in a second pass, so the
    // uncompressed library never exists on disk; which is the whole point when a shard is
    // gigabytes and the scratch volume is not.
    let compress = opts.out.ends_with(".gz");
    let writer_handle = thread::spawn(move || -> Result<LibraryStats> {
        let stream: Box<dyn Write> = if compress {
            Box::new(flate2::write::GzEncoder::new(
                writer_file,
                flate2::Compression::default(),
            ))
        } else {
            Box::new(writer_file)
        };
        let writer = BufWriter::new(stream);
        let mut sink: Box<dyn LibrarySink> = match format {
            LibraryFormat::DiannTsv => Box::new(DiannSink { writer }),
            LibraryFormat::MzSpecLib => {
                Box::new(crate::mzspeclib::MzSpecLibSink::new(writer, &out_path))
            }
        };
        sink.header(&header_config)?;
        let mut stats = stats;
        for result in result_rx {
            write_predicted_batch(sink.as_mut(), &mut stats, &result?, max_fragments)?;
        }
        sink.finish()?;
        Ok(stats)
    });

    let mut worker_handles = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let work_rx = Arc::clone(&work_rx);
        let result_tx = result_tx.clone();
        let artifact = Arc::clone(&artifact);
        let context = context.clone();
        let charges = charges.clone();
        let min_intensity = opts.min_intensity;
        worker_handles.push(thread::spawn(move || loop {
            let work = work_rx.lock().expect("work queue mutex poisoned").recv();
            match work {
                Ok(Some(batch)) => {
                    let result = predict_batch(&artifact, &context, &charges, min_intensity, batch);
                    if result_tx.send(result).is_err() {
                        break;
                    }
                }
                Ok(None) | Err(_) => break,
            }
        }));
    }

    let mut pending: BTreeMap<usize, Vec<PendingPeptide>> = BTreeMap::new();
    for (sequence, proteins) in peptides {
        let protein_group = proteins.into_iter().collect::<Vec<_>>().join(";");
        for (peptide, diann_sequence) in modified_forms(
            &sequence,
            &fixed_rules,
            &variable_rules,
            opts.max_variable_mods,
        )? {
            let length = peptide.sequence.len();
            let ready = {
                let bucket = pending.entry(length).or_default();
                bucket.push(PendingPeptide {
                    stripped: sequence.clone(),
                    protein_group: protein_group.clone(),
                    peptide,
                    diann_sequence,
                });
                (bucket.len() >= INFERENCE_BATCH_SIZE).then(|| std::mem::take(bucket))
            };
            if let Some(batch) = ready {
                work_tx
                    .send(Some(batch))
                    .context("sending inference batch")?;
            }
        }
    }
    for batch in pending.into_values().filter(|batch| !batch.is_empty()) {
        work_tx
            .send(Some(batch))
            .context("sending inference batch")?;
    }
    for _ in 0..worker_count {
        work_tx.send(None).context("stopping inference worker")?;
    }
    drop(work_tx);
    for handle in worker_handles {
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("inference worker panicked"))?;
    }
    drop(result_tx);
    let stats = writer_handle
        .join()
        .map_err(|_| anyhow::anyhow!("library writer panicked"))??;
    if let Some(path) = opts.config_out {
        write_config(path, &config, &stats)?;
    }
    Ok(stats)
}

/// blake2b-256 of a file's bytes, hex, streamed so a multi-GB input is not held in memory.
///
/// Identity rather than integrity: two libraries generated from the same digests and the same
/// settings are the same library, which is what makes a published one reproducible.
fn file_digest(path: &str) -> Result<String> {
    use blake2::digest::{Update, VariableOutput};
    let mut file = File::open(path).with_context(|| format!("hashing {path}"))?;
    let mut hasher = blake2::Blake2bVar::new(32).expect("32 is a valid blake2b output length");
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = std::io::Read::read(&mut file, &mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let mut out = [0u8; 32];
    hasher.finalize_variable(&mut out).expect("32-byte output");
    Ok(out.iter().map(|byte| format!("{byte:02x}")).collect())
}

/// Resolve everything that determined the library's bytes.
///
/// The digests of the two inputs and every knob as resolved, not as typed: defaults are recorded
/// explicitly. Counts are not here; they are only known once the library is written; so
/// [`write_config`] appends them. A library whose provenance lives only in a shell history is one
/// nobody can regenerate or trust, which is why this value is both embedded (mzSpecLib) and
/// written beside the library (sidecar) rather than reassembled per format.
fn resolve_config(
    opts: &LibraryOptions<'_>,
    format: LibraryFormat,
    artifact: &Artifact,
    model_digest: &str,
) -> Result<serde_json::Value> {
    let ms_context = match opts.ms_context {
        Some(pepdistill_core::MsContext::Named(name)) => serde_json::json!({"setup": name}),
        Some(pepdistill_core::MsContext::Factors {
            instrument,
            detector,
            fragmentation,
            energy,
        }) => serde_json::json!({
            "instrument": instrument,
            "detector": detector,
            "fragmentation": fragmentation,
            "energy": energy,
        }),
        None => serde_json::Value::Null,
    };
    // What the retention numbers in this library actually are. The normalized value is an index,
    // not a duration, even though the vocabulary makes us declare it in minutes; so the file
    // says so in plain text rather than leaving a reader to infer it from a unit that cannot be
    // right.
    //
    // Which index is a property of the corpus the model trained on, so the scale is stated as the
    // convention that defines it and then checked: predicting the two anchors says whether this
    // artifact is on that scale. An artifact from another corpus reports `on_scale: false` and its
    // own numbers rather than inheriting a claim that happens to be true of ours.
    let anchors = pepdistill_core::landmarks::check_retention_scale(artifact)?;
    let normalized_retention = serde_json::json!({
        "term": "MS:1000896|normalized retention time",
        "kind": "dimensionless index, minutes-like",
        "scale": pepdistill_core::landmarks::SCALE_DESCRIPTION,
        "anchor_check": {
            "on_scale": anchors.on_scale(),
            "max_abs_error": anchors.max_abs_error,
            "anchors": anchors
                .anchors
                .iter()
                .map(|(peptide, expected, predicted)| serde_json::json!({
                    "peptide": peptide,
                    "expected": expected,
                    "predicted": predicted,
                }))
                .collect::<Vec<_>>(),
        },
    });
    let raw_retention = match opts.chrom_context {
        Some(name) => serde_json::json!({
            "term": "MS:1000894|retention time",
            "unit": "minute",
            "chrom_context": name,
        }),
        None => serde_json::Value::Null,
    };
    Ok(serde_json::json!({
        "generator": {
            "tool": "pepdistill-cli library",
            "version": env!("CARGO_PKG_VERSION"),
        },
        "inputs": {
            // The spec as asked for, which for a bundled model is a name rather than a path. A
            // temp path is not an identity; the digest is, and it is computed from the bytes that
            // were actually loaded either way.
            "model": opts.model,
            "model_blake2b_256": model_digest,
            "fasta": opts.fasta,
            "fasta_blake2b_256": file_digest(opts.fasta)?,
        },
        "digestion": {
            "enzyme": "trypsin",
            "missed_cleavages": opts.missed_cleavages,
            "min_length": opts.min_length,
            "max_length": opts.max_length,
            "min_charge": opts.min_charge,
            "max_charge": opts.max_charge,
        },
        "modifications": {
            "fixed": opts.fixed_mods,
            "variable": opts.variable_mods,
            "max_variable_mods": opts.max_variable_mods,
        },
        "context": {
            "ms": ms_context,
            "chrom": opts.chrom_context,
        },
        "retention": {
            "normalized": normalized_retention,
            "raw": raw_retention,
        },
        "fragments": {
            "min_intensity": opts.min_intensity,
            "max_fragments": opts.max_fragments,
        },
        "output": {
            "path": opts.out,
            "format": format.name(),
            "compressed": opts.out.ends_with(".gz"),
        },
    }))
}

/// Write the resolved configuration, plus the counts that came out, beside the library.
fn write_config(path: &str, config: &serde_json::Value, stats: &LibraryStats) -> Result<()> {
    let mut config = config.clone();
    let output = config["output"]
        .as_object_mut()
        .expect("resolve_config writes an object under `output`");
    output.insert("proteins".into(), stats.proteins.into());
    output.insert("peptides".into(), stats.peptides.into());
    output.insert("precursors".into(), stats.precursors.into());
    output.insert("fragments".into(), stats.fragments.into());
    std::fs::write(path, serde_json::to_string_pretty(&config)? + "\n")
        .with_context(|| format!("writing {path}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tryptic_digest_honors_proline_and_missed_cleavages() {
        assert_eq!(
            digest_tryptic("AKPEPTIDERAAK", 0, 1, 30),
            vec!["AKPEPTIDER", "AAK"]
        );
        assert_eq!(
            digest_tryptic("AKPEPTIDERAAK", 1, 1, 30),
            vec!["AKPEPTIDER", "AKPEPTIDERAAK", "AAK"]
        );
    }

    #[test]
    fn modifications_render_for_model_and_diann() {
        let fixed = vec![parse_modification_rule("C[UNIMOD:4]").unwrap()];
        let variable = vec![parse_modification_rule("M[UNIMOD:35]").unwrap()];
        let forms = modified_forms("ACDM", &fixed, &variable, 1).unwrap();
        let (peptide, diann) = &forms[1];
        assert_eq!(peptide.modified_sequence(), "AC[UNIMOD:4]DM[UNIMOD:35]");
        assert_eq!(diann, "AC(UniMod:4)DM(UniMod:35)");
        assert_eq!(peptide.mods[1].0, Site::Residue(3));
    }

    #[test]
    fn enumerates_cyspat_phospho_and_oxidation_with_one_global_cap() {
        let variable = [
            parse_modification_rule("C[UNIMOD:2057]").unwrap(),
            parse_modification_rule("STY[UNIMOD:21]").unwrap(),
            parse_modification_rule("M[UNIMOD:35]").unwrap(),
        ];
        let forms = modified_forms("ACSM", &[], &variable, 3).unwrap();
        // Three independently eligible sites -> C(3,0)+C(3,1)+C(3,2)+C(3,3).
        assert_eq!(forms.len(), 8);
        assert!(forms.iter().any(|(peptide, diann)| {
            peptide.modified_sequence() == "AC[UNIMOD:2057]S[UNIMOD:21]M[UNIMOD:35]"
                && diann == "AC(UniMod:2057)S(UniMod:21)M(UniMod:35)"
        }));
    }

    #[test]
    fn overlapping_fixed_and_variable_targets_are_detectable() {
        let fixed = parse_modification_rule("C[UNIMOD:4]").unwrap();
        let variable = parse_modification_rule("C[UNIMOD:2057]").unwrap();
        assert!(target_overlap(&fixed.target, &variable.target));
    }

    #[test]
    fn output_suffix_picks_the_format_through_compression() {
        for path in ["lib.mzspeclib.txt", "lib.mzspeclib.txt.gz", "lib.mzspeclib"] {
            assert_eq!(
                LibraryFormat::for_output(path),
                LibraryFormat::MzSpecLib,
                "{path}"
            );
        }
        for path in ["lib.tsv", "lib.tsv.gz", "lib", "mzspeclib.tsv"] {
            assert_eq!(
                LibraryFormat::for_output(path),
                LibraryFormat::DiannTsv,
                "{path}"
            );
        }
    }

    #[test]
    fn ccs_conversion_matches_alphabase_formula() {
        let mobility = ccs_to_bruker_mobility(400.0, 2, 500.0);
        assert!((mobility - 0.985056867).abs() < 1e-8, "{mobility}");
    }
}
