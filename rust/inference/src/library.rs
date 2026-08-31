use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use msspeculator_core::peptide::{ModSpec, Peptide, Site};
use msspeculator_core::proforma::{parse_modification_rule, ModificationRule, ModificationTarget};
use msspeculator_core::{
    predict_peptide_batch_charges_prepared, Artifact, ModelSource, MsContext, Prediction,
    PreparedContext,
};

use crate::progress::{Phase, ProgressFn, Reporter};
use crate::proteome::{peptides, Digest, DigestRules, PeptideRef, ProteinGroup, Residues};
use crate::provenance::{resolve_provenance, write_sidecar, LibraryProvenance, Output};

const CCS_IM_COEF: f64 = 1059.62245;
const IM_GAS_MASS: f64 = 28.0;
const INFERENCE_BATCH_SIZE: usize = 64;

/// Everything that decides what the library *contains*, with nothing about where it goes.
///
/// Split from [`LibraryOptions`] so a caller with its own [`LibrarySink`] can generate a library
/// without inventing an output path to satisfy fields it will never use. See [`stream_library`].
pub struct StreamOptions<'a> {
    /// Built-in or file-backed model. The CLI converts its string option before calling this API.
    pub model: ModelSource,
    pub fasta: &'a Path,
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
    /// Add pseudo-reversed peptide decoys. A decoy whose stripped sequence is already a target is
    /// skipped; two decoys cannot collide with each other, since reversing is an involution.
    pub generate_decoys: bool,
    /// Called as the build advances, or `None` to report nothing.
    ///
    /// Bounded, but not tied to any clock: a build reports thousands of times rather than
    /// millions, and each phase reports both of its ends. A callback that writes a log line wants
    /// its own throttle. [`Progress`](crate::Progress) says what each phase counts.
    pub progress: Option<&'a ProgressFn<'a>>,
}

impl StreamOptions<'_> {
    /// The settings that can be refused without reading anything, checked on the type that holds
    /// them so a caller can ask before committing to a run.
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.min_intensity) {
            bail!(
                "min_intensity must be in [0, 1], got {}",
                self.min_intensity
            );
        }
        if self.min_charge < 1 || self.max_charge < self.min_charge {
            bail!(
                "invalid charge range {}..={}",
                self.min_charge,
                self.max_charge
            );
        }
        if self.min_length < 2 || self.max_length < self.min_length {
            bail!(
                "invalid peptide length range {}..={}",
                self.min_length,
                self.max_length
            );
        }
        Ok(())
    }
}

/// Generating a library and writing it to a file.
///
/// The file is what makes the extra fields meaningful: the path picks the format and the
/// compression, and names the sidecar's subject. A caller supplying its own sink has none of
/// those questions to answer and wants [`StreamOptions`] instead.
pub struct LibraryOptions<'a> {
    pub stream: StreamOptions<'a>,
    pub out: &'a Path,
    /// Where to write the resolved-configuration sidecar, or `None` to skip it.
    pub config_out: Option<&'a Path>,
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
enum LibraryFormat {
    DiannTsv,
    MzSpecLib,
}

impl LibraryFormat {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::DiannTsv => "diann-tsv",
            Self::MzSpecLib => "mzspeclib-text",
        }
    }
}

/// Both things the output path decides: which writer, and whether to compress.
///
/// One function because they are one question about one suffix; answering them separately let
/// `.gz` be detected two ways that disagreed on `--out .gz`. Read lossily rather than requiring
/// UTF-8: the suffixes are ASCII, and lossy replacement only ever touches bytes above 0x7f, so a
/// suffix match on the lossy form is correct for any path the OS will accept.
fn output_spelling(path: &Path) -> (LibraryFormat, bool) {
    let text = path.to_string_lossy();
    let compressed = text.ends_with(".gz");
    let stem = text.strip_suffix(".gz").unwrap_or(&text);
    let format = if stem.ends_with(".mzspeclib.txt") || stem.ends_with(".mzspeclib") {
        LibraryFormat::MzSpecLib
    } else {
        LibraryFormat::DiannTsv
    };
    (format, compressed)
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
    /// The peptide's residues. A view over the digest rather than a string, so a decoy is the
    /// same residues read with its interior reversed instead of a second copy of them.
    pub stripped: Residues<'a>,
    /// The proteins whose digest produced this peptide, as a list. DIA-NN's `;` separator is a
    /// property of that format alone, so its writer joins them and nothing else has to know.
    pub proteins: ProteinGroup<'a>,
    /// The peptidoform: residues plus normalized modification sites. Both output formats derive
    /// their own spelling from it, so the row carries neither one's.
    pub peptide: &'a Peptide,
    /// ProForma spelling of the same peptidoform, e.g. `PEPC[UNIMOD:4]IDER`, as the model
    /// returned it.
    pub proforma: &'a str,
    /// Whether this spectrum belongs to the generated decoy set.
    pub decoy: bool,
    /// Stable ID for a target/decoy sequence pair. The target keeps the ID when its decoy is
    /// collision-skipped; the ID is absent when decoys are disabled.
    pub decoy_pair_id: Option<usize>,
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
///
/// `Send` because prediction and serialization run on separate threads: workers predict while the
/// sink writes, so the sink is moved into the writer thread. A sink that cannot cross that move
/// would serialize the two halves of the job against each other.
pub trait LibrarySink: Send {
    /// Called once, before any spectrum, with what produced this library. A format that has
    /// nowhere to record provenance ignores the argument; one that does can embed it whole.
    fn header(&mut self, provenance: &LibraryProvenance) -> Result<()>;
    fn spectrum(&mut self, row: &SpectrumRow<'_>) -> Result<()>;
    /// Flush the stream. Called once, after the last spectrum: a buffered library that is never
    /// flushed is a truncated library.
    fn finish(&mut self) -> Result<()>;
}

#[derive(Debug, Default)]
pub struct LibraryStats {
    pub proteins: usize,
    pub peptides: usize,
    pub precursors: usize,
    pub fragments: usize,
    /// Number of decoy precursor spectra written.
    pub decoys: usize,
    /// Wall time spent reading and digesting the FASTA.
    ///
    /// Split from the rest because the two halves scale on different things: digestion grows
    /// with the proteome, prediction with the precursor count and the model. A build that got
    /// slower is a different problem depending on which number moved.
    pub digest: Duration,
    /// Wall time spent reading the model and preparing its context.
    ///
    /// Its own number rather than part of `predict`, because it scales on neither of the things
    /// the other two scale on — it scales on the artifact and on whether the page cache is cold.
    /// Folded into `predict`, a rebuild that got four minutes slower could not say whether the
    /// model took longer to load or longer to run.
    pub load: Duration,
    /// Wall time from the end of loading to the last spectrum handed to the sink.
    pub predict: Duration,
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

/// The decoy peptidoform: residues pseudo-reversed, and each residue modification carried to
/// wherever its residue ended up. Terminal modifications stay terminal.
fn pseudo_reverse_peptide(peptide: &Peptide) -> Peptide {
    let length = peptide.sequence.chars().count();
    let mods = peptide
        .mods
        .iter()
        .map(|(site, spec)| {
            let site = match site {
                Site::Residue(index) if *index > 0 && *index + 1 < length => {
                    Site::Residue(length - 1 - index)
                }
                _ => *site,
            };
            (site, spec.clone())
        })
        .collect();
    // The same reversal `SpectrumRow.stripped` reports and the collision check asks about, rather
    // than a second implementation of the rule: a divergence between them would emit a library
    // whose stripped column disagreed with its modified-peptide column, silently.
    Peptide::new(
        Residues::pseudo_reversed(&peptide.sequence).to_string(),
        mods,
    )
}

fn modified_forms(
    sequence: &str,
    fixed_rules: &[ModificationRule],
    variable_rules: &[ModificationRule],
    max_variable: usize,
) -> Result<Vec<Peptide>> {
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
            Ok(peptide)
        })
        .collect()
}

fn ccs_to_bruker_mobility(ccs: f64, charge: i64, precursor_mz: f64) -> f64 {
    let mass = precursor_mz * charge as f64;
    let reduced_mass = mass * IM_GAS_MASS / (mass + IM_GAS_MASS);
    ccs * reduced_mass.sqrt() / charge as f64 / CCS_IM_COEF
}

/// One peptidoform on its way to a worker.
///
/// Carries the flyweight rather than copies: `source` is an atomic increment and an index, and it
/// answers the stripped sequence and the protein group without materializing either. A decoy adds
/// nothing to that, because `decoy` already says how to read the same residues.
struct PendingPeptide {
    source: PeptideRef,
    peptide: Peptide,
    decoy: bool,
    decoy_pair_id: Option<usize>,
}

struct PredictedPeptide {
    source: PeptideRef,
    /// Kept past prediction because it is what each format spells its own way; carrying a
    /// rendered string instead would put one format's spelling on a format-neutral record.
    peptide: Peptide,
    predictions: Vec<Prediction>,
    decoy: bool,
    decoy_pair_id: Option<usize>,
}

impl PredictedPeptide {
    fn stripped(&self) -> Residues<'_> {
        if self.decoy {
            Residues::pseudo_reversed(self.source.residues())
        } else {
            Residues::target(self.source.residues())
        }
    }
}

/// Validate one prediction and cap its transition list.
fn spectrum_row<'a>(
    item: &'a PredictedPeptide,
    prediction: &'a Prediction,
    max_fragments: Option<usize>,
) -> Result<SpectrumRow<'a>> {
    // The identifier every refusal below names. ProForma rather than a format's own spelling, so
    // an error message does not speak DIA-NN at someone writing mzSpecLib.
    let proforma = prediction.peptide.as_str();
    let charge = prediction.charge;
    if !prediction.precursor_mz.is_finite()
        || !prediction.rt.is_finite()
        || !prediction.ccs.is_finite()
        || prediction.precursor_mz <= 0.0
        || prediction.ccs <= 0.0
    {
        bail!(
            "non-physical precursor prediction for {} charge {}: mz={}, rt={}, ccs={}",
            proforma,
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
            proforma,
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
                proforma,
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
        stripped: item.stripped(),
        proteins: item.source.proteins(item.decoy),
        peptide: &item.peptide,
        proforma,
        decoy: item.decoy,
        decoy_pair_id: item.decoy_pair_id,
        charge,
        precursor_mz: prediction.precursor_mz,
        neutral_mass: (prediction.precursor_mz - msspeculator_core::chem::PROTON) * charge as f64,
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
        .map(|item| ((item.source, item.decoy, item.decoy_pair_id), item.peptide))
        .unzip();
    let predictions = predict_peptide_batch_charges_prepared(
        artifact,
        &peptides,
        charges,
        context,
        min_intensity,
    )?;
    // The peptides are borrowed by the call above rather than consumed, so each one survives to
    // be handed on: a sink spells its own peptidoform, and nothing has to be rendered twice.
    Ok(metadata
        .into_iter()
        .zip(peptides)
        .zip(predictions)
        .map(
            |(((source, decoy, decoy_pair_id), peptide), predictions)| PredictedPeptide {
                source,
                peptide,
                predictions,
                decoy,
                decoy_pair_id,
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
            stats.decoys += usize::from(row.decoy);
            stats.fragments += row.peaks.len();
            sink.spectrum(&row)?;
        }
    }
    Ok(())
}

/// Bucket one peptidoform by length, dispatching the bucket once it is a full batch.
///
/// A send failure means every worker has gone, which happens when the writer thread has already
/// failed. Reported as `false` rather than an error, so the producer can stop and let the join
/// surface the real one instead of this symptom of it.
fn queue_pending(
    pending: &mut BTreeMap<usize, Vec<PendingPeptide>>,
    work_tx: &mpsc::SyncSender<Option<Vec<PendingPeptide>>>,
    item: PendingPeptide,
) -> bool {
    let length = item.peptide.sequence.len();
    let ready = {
        let bucket = pending.entry(length).or_default();
        bucket.push(item);
        (bucket.len() >= INFERENCE_BATCH_SIZE).then(|| std::mem::take(bucket))
    };
    match ready {
        Some(batch) => work_tx.send(Some(batch)).is_ok(),
        None => true,
    }
}

/// Generate a library and hand every spectrum to `sink`, writing no file.
///
/// The counterpart to [`write_library`] for a caller that has somewhere better to put the rows
/// than a file it would immediately parse back. The sink sees the same [`SpectrumRow`] values the
/// bundled DIA-NN and mzSpecLib writers see, already validated and capped, and its `header`
/// receives the provenance, so a caller keeps it without the sidecar.
pub fn stream_library(
    opts: &StreamOptions<'_>,
    sink: impl LibrarySink + 'static,
) -> Result<LibraryStats> {
    run_library(opts, None, || Ok(Box::new(sink))).map(|(stats, _)| stats)
}

/// Generate a library and write it to `opts.out`, picking the format from the path.
pub fn write_library(opts: &LibraryOptions<'_>) -> Result<LibraryStats> {
    let (format, compressed) = output_spelling(opts.out);
    let out_path = opts.out.to_path_buf();
    // Passed as a thunk rather than a ready-made sink: creating the file truncates it, and every
    // check in `run_library` happens before this runs. A typo in a regeneration command must not
    // destroy the library it was meant to replace.
    let make_sink = move || -> Result<Box<dyn LibrarySink>> {
        let file = File::create(&out_path)
            .with_context(|| format!("creating library {}", out_path.display()))?;
        // A `.gz` suffix compresses in the writer thread rather than in a second pass, so the
        // uncompressed library never exists on disk; which is the whole point when a shard is
        // gigabytes and the scratch volume is not.
        let stream: Box<dyn Write + Send> = if compressed {
            Box::new(flate2::write::GzEncoder::new(
                file,
                flate2::Compression::default(),
            ))
        } else {
            Box::new(file)
        };
        let writer = BufWriter::new(stream);
        Ok(match format {
            LibraryFormat::DiannTsv => Box::new(crate::diann::DiannSink { writer }),
            LibraryFormat::MzSpecLib => {
                Box::new(crate::mzspeclib::MzSpecLibSink::new(writer, &out_path))
            }
        })
    };
    let output = Output {
        path: opts.out.display().to_string(),
        format: format.name(),
        compressed,
        counts: None,
        timing: None,
    };
    let (stats, provenance) = run_library(&opts.stream, Some(output), make_sink)?;
    if let Some(path) = opts.config_out {
        write_sidecar(path, &provenance, &stats)?;
    }
    Ok(stats)
}

/// The whole job, with the output half optional.
///
/// Returns the provenance alongside the counts because the sidecar needs both, and resolving it
/// twice would let the copy inside the library disagree with the copy beside it.
fn run_library(
    opts: &StreamOptions<'_>,
    output: Option<Output>,
    make_sink: impl FnOnce() -> Result<Box<dyn LibrarySink>>,
) -> Result<(LibraryStats, LibraryProvenance)> {
    opts.validate()?;
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
    let reporter = Reporter::new(opts.progress);
    let digest_started = Instant::now();
    let digest = Arc::new(Digest::read(
        opts.fasta,
        &DigestRules {
            missed_cleavages: opts.missed_cleavages,
            min_length: opts.min_length,
            max_length: opts.max_length,
        },
        &reporter,
    )?);
    if digest.is_empty() {
        bail!("FASTA digest produced no peptides");
    }
    let digest_elapsed = digest_started.elapsed();
    // The denominator for the rest of the build. Peptides rather than precursors: a peptide's
    // precursor count depends on how many modified forms it turns out to have, which is only
    // known once it is enumerated, and a total that is still being discovered is no total at all.
    let total_peptides = digest.peptides() as u64;
    // Announced, not measured: reading an artifact is one call that reports nothing on the way
    // through, so a zero total is the truthful spelling and a renderer shows the label alone.
    reporter.at(Phase::Loading, 0, 0);
    // Clocked separately for the same reason the other two are clocked separately: loading scales
    // on the artifact and on whether the page cache is cold, neither of which has anything to do
    // with the proteome or the precursor count. Billed to the phase the callback already named,
    // so the three durations are disjoint and sum to the build.
    let load_started = Instant::now();
    let model = msspeculator_core::load_source(opts.model.clone())?;
    let mut artifact = model.artifact;
    apply_activation_override(&mut artifact, opts.activation)?;
    let context = PreparedContext::new(&artifact, opts.ms_context, opts.chrom_context)?;
    let load_elapsed = load_started.elapsed();
    let predict_started = Instant::now();
    // The opening update, before the threads start and before provenance hashes the inputs. The
    // in-loop report below only fires once a peptide has been enumerated, so a build small enough
    // to finish inside one report interval would otherwise go from invisible to 100% in one step.
    reporter.at(Phase::Predicting, 0, total_peptides);
    let charges = (opts.min_charge..=opts.max_charge).collect::<Vec<_>>();
    let stats = LibraryStats {
        proteins: digest.proteins(),
        peptides: digest.peptides(),
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
    // Resolved before the first spectrum because an mzSpecLib header carries it, and a header is
    // the first thing on the stream. The sidecar reuses the same value, so the copy bundled in
    // the library and the copy beside it are the same copy.
    let provenance = resolve_provenance(opts, output, &artifact, &model.digest)?;
    // Everything that can fail on the way in has now run, so this is the first thing to touch the
    // filesystem. The header goes out here too, on this thread: it is the first bytes on the
    // stream either way, and writing it before the spawn means a broken header is an error from
    // this call rather than a panic reported by a joined thread.
    let mut sink = make_sink()?;
    sink.header(&provenance)?;
    let writer_handle = thread::spawn(move || -> Result<LibraryStats> {
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
    // The producer keeps only the sender. Holding a receiver here too would mean `send` never
    // reports a disconnect, so a writer that failed would leave every worker gone, the queue
    // undrained, and this thread blocked on a full channel forever: a build that hangs silently
    // instead of reporting the error it already has.
    drop(work_rx);

    let mut next_decoy_pair_id = 1usize;
    let mut pending: BTreeMap<usize, Vec<PendingPeptide>> = BTreeMap::new();
    let mut peptides_done = 0u64;
    // Set when the workers have gone, which only happens once the writer has failed. The producer
    // stops enumerating and falls through to the joins, where that failure is waiting.
    let mut consumers_gone = false;
    for source in peptides(&digest) {
        let sequence = source.residues();
        let target_forms = modified_forms(
            sequence,
            &fixed_rules,
            &variable_rules,
            opts.max_variable_mods,
        )?;
        // Whether this peptide contributes a decoy, decided without building one.
        //
        // Only the target collision has to be checked. Pseudo-reversal pins the termini and turns
        // the interior around, so applying it twice returns the original: it is an involution,
        // therefore injective, therefore two distinct peptides can never reverse to the same
        // decoy. `pseudo_reverse_is_an_involution` pins that, and is what a change of decoy
        // method would break first.
        let emit_decoy =
            opts.generate_decoys && !digest.contains(Residues::pseudo_reversed(sequence));
        // Assign the ID even when the decoy is skipped, so consumers can distinguish a target-only
        // collision group from a library generated with decoys disabled.
        let decoy_pair_id = opts.generate_decoys.then(|| {
            let pair_id = next_decoy_pair_id;
            next_decoy_pair_id += 1;
            pair_id
        });
        for peptide in target_forms {
            consumers_gone |= !queue_pending(
                &mut pending,
                &work_tx,
                PendingPeptide {
                    source: source.clone(),
                    peptide: peptide.clone(),
                    decoy: false,
                    decoy_pair_id,
                },
            );
            if emit_decoy {
                consumers_gone |= !queue_pending(
                    &mut pending,
                    &work_tx,
                    PendingPeptide {
                        source: source.clone(),
                        peptide: pseudo_reverse_peptide(&peptide),
                        decoy: true,
                        decoy_pair_id,
                    },
                );
            }
        }
        peptides_done += 1;
        // Held below the total, because the producer runs ahead of the workers: it knows what has
        // been enumerated, never what has been written. Only the update after the join is entitled
        // to say the phase is done.
        reporter.at(
            Phase::Predicting,
            peptides_done.min(total_peptides - 1),
            total_peptides,
        );
        if consumers_gone {
            break;
        }
    }
    if !consumers_gone {
        for batch in pending.into_values().filter(|batch| !batch.is_empty()) {
            if work_tx.send(Some(batch)).is_err() {
                break;
            }
        }
        for _ in 0..worker_count {
            if work_tx.send(None).is_err() {
                break;
            }
        }
    }
    drop(work_tx);
    for handle in worker_handles {
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("inference worker panicked"))?;
    }
    drop(result_tx);
    let mut stats = writer_handle
        .join()
        .map_err(|_| anyhow::anyhow!("library writer panicked"))??;
    stats.digest = digest_elapsed;
    stats.load = load_elapsed;
    stats.predict = predict_started.elapsed();
    // The closing update, after the join rather than after the last peptide was queued: the
    // producer runs a bounded distance ahead of the workers, so a bar that reached 100% when
    // enumeration ended would claim the library was finished while it was still being written.
    reporter.at(Phase::Predicting, total_peptides, total_peptides);
    Ok((stats, provenance))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::progress::Progress;
    use crate::scratch::Scratch;

    /// What an embedding caller looks like: keeps the rows, writes nothing.
    ///
    /// Rows borrow from the prediction they came out of, so anything retained past `spectrum`
    /// has to be copied. Shared behind a mutex rather than a channel: the sink is moved onto the
    /// writer thread, so this exercises the `Send` bound either way, and the test does not have
    /// to keep receivers alive to stop the sink panicking mid-run.
    #[derive(Default)]
    struct Collected {
        rows: Vec<(String, i64, usize)>,
        proteins: Vec<String>,
        model: Option<String>,
        output_present: Option<bool>,
        finished: usize,
    }

    struct CollectingSink(Arc<Mutex<Collected>>);

    impl LibrarySink for CollectingSink {
        fn header(&mut self, provenance: &LibraryProvenance) -> Result<()> {
            let mut collected = self.0.lock().expect("collector mutex poisoned");
            // Asserted through the typed provenance, not through JSON: reaching back through a
            // `Value` here would be reaching past the interface this whole split exists to give.
            collected.model = Some(provenance.inputs.model.clone());
            collected.output_present = Some(provenance.output.is_some());
            Ok(())
        }

        fn spectrum(&mut self, row: &SpectrumRow<'_>) -> Result<()> {
            let mut collected = self.0.lock().expect("collector mutex poisoned");
            collected
                .rows
                .push((row.proforma.to_string(), row.charge, row.peaks.len()));
            collected.proteins = row.proteins.iter().map(|id| id.to_string()).collect();
            Ok(())
        }

        fn finish(&mut self) -> Result<()> {
            self.0.lock().expect("collector mutex poisoned").finished += 1;
            Ok(())
        }
    }

    fn tiny_fasta() -> Scratch {
        let scratch = Scratch::new("tiny.fasta");
        std::fs::write(scratch.path(), ">protein_one description\nPEPTIDEMR\n").unwrap();
        scratch
    }

    fn stream_options(fasta: &Path) -> StreamOptions<'_> {
        StreamOptions {
            model: ModelSource::Builtin(msspeculator_core::BuiltinModel::SmallV0),
            fasta,
            activation: None,
            ms_context: None,
            chrom_context: None,
            min_intensity: 0.01,
            missed_cleavages: 0,
            min_length: 9,
            max_length: 9,
            min_charge: 2,
            max_charge: 2,
            fixed_mods: &[],
            variable_mods: &[],
            max_variable_mods: 0,
            max_fragments: None,
            generate_decoys: false,
            progress: None,
        }
    }

    /// A FASTA big enough that both phases report while they run rather than only at their close.
    ///
    /// Sized past the reporter's stride on purpose: with a handful of peptides every intermediate
    /// update is suppressed, and a progress test that only ever sees a closing update cannot fail
    /// for the reason it exists.
    fn several_proteins() -> Scratch {
        let scratch = Scratch::new("several.fasta");
        let mut text = String::new();
        // Each digests to itself at the fixture's fixed length of 9: no internal cleavage site,
        // so a protein contributes exactly one peptide and the totals stay readable.
        for i in 0..300u32 {
            let sequence: String = format!("{i:04}")
                .bytes()
                .map(|digit| b"GASPVTCLIN"[usize::from(digit - b'0')] as char)
                .chain("PTIDR".chars())
                .collect();
            text.push_str(&format!(">protein_{i} description\n{sequence}\n"));
        }
        std::fs::write(scratch.path(), text).unwrap();
        scratch
    }

    /// What a caller needs from the callback: the phases in order, each one monotone, and a
    /// closing update that only lands once the library really is finished.
    #[test]
    fn progress_walks_the_phases_in_order_and_ends_at_the_total() {
        let fasta = several_proteins();
        let seen = Mutex::new(Vec::new());
        let report = |progress: Progress| seen.lock().unwrap().push(progress);
        let mut opts = stream_options(fasta.path());
        opts.progress = Some(&report);

        let stats = stream_library(
            &opts,
            CollectingSink(Arc::new(Mutex::new(Collected::default()))),
        )
        .unwrap();

        let seen = seen.into_inner().unwrap();
        let order: Vec<Phase> = seen.iter().map(|p| p.phase).collect();
        let first = |phase: Phase| order.iter().position(|&p| p == phase).unwrap();
        assert!(first(Phase::Digesting) < first(Phase::Loading));
        assert!(first(Phase::Loading) < first(Phase::Predicting));

        for phase in [Phase::Digesting, Phase::Loading, Phase::Predicting] {
            let updates: Vec<&Progress> = seen.iter().filter(|p| p.phase == phase).collect();
            assert!(!updates.is_empty(), "{phase:?} never reported");
            // The opening end. Without it a bar first appears partway along, or — for a phase
            // short enough to finish inside one report interval — goes from invisible to 100%.
            assert_eq!(updates[0].done, 0, "{phase:?} did not report its start");
            assert!(
                updates.windows(2).all(|w| w[0].done <= w[1].done),
                "{phase:?} went backwards"
            );
            let last = updates.last().unwrap();
            assert_eq!(last.done, last.total, "{phase:?} did not finish");
        }

        // Digestion is billed in bytes of the file it read, prediction in the peptides that came
        // out of it, so the totals are checkable against something other than themselves.
        let digesting = seen.iter().find(|p| p.phase == Phase::Digesting).unwrap();
        assert_eq!(
            digesting.total,
            std::fs::metadata(fasta.path()).unwrap().len()
        );
        let predicting: Vec<&Progress> = seen
            .iter()
            .filter(|p| p.phase == Phase::Predicting)
            .collect();
        assert_eq!(predicting[0].total, stats.peptides as u64);
        // The point of the phase, and what a single closing update would not prove: it reports
        // while the build runs.
        assert!(
            predicting.len() > 1,
            "predicting reported only its closing update"
        );
        // Pins the claim that the closing update lands after the writer joins rather than when
        // enumeration ended, which is otherwise only a comment.
        assert!(
            predicting[..predicting.len() - 1]
                .iter()
                .all(|p| p.done < p.total),
            "an update claimed completion before the writer joined"
        );
        // Loading measures nothing, so it says so rather than inventing a denominator.
        let loading = seen.iter().find(|p| p.phase == Phase::Loading).unwrap();
        assert_eq!((loading.done, loading.total), (0, 0));
    }

    #[test]
    fn a_build_records_how_long_each_phase_took() {
        let fasta = tiny_fasta();
        let stats = stream_library(
            &stream_options(fasta.path()),
            CollectingSink(Arc::new(Mutex::new(Collected::default()))),
        )
        .unwrap();
        assert!(stats.predict > std::time::Duration::ZERO);
        // Reading an artifact is the phase that exists to explain a pause, so it is the one whose
        // duration has to be its own number rather than folded into prediction's.
        assert!(stats.load > std::time::Duration::ZERO);
        assert!(stats.load < stats.load + stats.predict);
        // Digesting one nine-residue protein can land inside a clock tick, so the claim is that
        // the three are recorded separately, not that any of them is large.
        assert!(stats.digest < stats.digest + stats.load + stats.predict);
    }

    /// A failing sink has to surface as an error, not as a hang.
    ///
    /// When the writer thread fails it drops the result channel, every worker breaks, and nothing
    /// drains the work queue. If the producer still held a receiver, its `send` would never report
    /// the disconnect and it would block on a full channel forever — a build stuck silently with
    /// the real error already in hand. The fixture has to outrun the queue for that to bite, which
    /// is why the small ones miss it.
    #[test]
    fn a_sink_that_fails_stops_the_build_instead_of_hanging_it() {
        struct FailingSink;

        impl LibrarySink for FailingSink {
            fn header(&mut self, _: &LibraryProvenance) -> Result<()> {
                Ok(())
            }

            fn spectrum(&mut self, _: &SpectrumRow<'_>) -> Result<()> {
                bail!("no space left on device")
            }

            fn finish(&mut self) -> Result<()> {
                Ok(())
            }
        }

        let scratch = Scratch::new("crowded.fasta");
        let mut text = String::new();
        for i in 0..6000u32 {
            let sequence: String = format!("{i:05}")
                .bytes()
                .map(|digit| b"GASPVTCLIN"[usize::from(digit - b'0')] as char)
                .chain("PTIDR".chars())
                .collect();
            text.push_str(&format!(">protein_{i} d\n{sequence}\n"));
        }
        std::fs::write(scratch.path(), text).unwrap();

        let mut opts = stream_options(scratch.path());
        opts.min_length = 10;
        opts.max_length = 10;
        let error = stream_library(&opts, FailingSink).unwrap_err();
        assert!(
            error.to_string().contains("no space left"),
            "the sink's own error should survive: {error}"
        );
    }

    /// A FASTA identifier is free to contain a semicolon, and DIA-NN's separator is a property of
    /// that one format. Nothing but its writer may assume otherwise.
    #[test]
    fn a_semicolon_in_an_identifier_survives_to_a_sink() {
        let scratch = Scratch::new("semicolon.fasta");
        std::fs::write(scratch.path(), ">weird;name desc\nPEPTIDEMR\n").unwrap();
        let collected = Arc::new(Mutex::new(Collected::default()));
        stream_library(
            &stream_options(scratch.path()),
            CollectingSink(Arc::clone(&collected)),
        )
        .unwrap();
        assert_eq!(collected.lock().unwrap().proteins, vec!["weird;name"]);
    }

    /// The point of the split: a library reaches a caller's own sink with no file anywhere, and
    /// the two entry points do not drift into producing different libraries.
    #[test]
    fn streaming_and_writing_produce_the_same_library() {
        let fasta = tiny_fasta();
        let collected = Arc::new(Mutex::new(Collected::default()));

        let streamed = stream_library(
            &stream_options(fasta.path()),
            CollectingSink(Arc::clone(&collected)),
        )
        .unwrap();

        let out = Scratch::new("library.tsv");
        let written = write_library(&LibraryOptions {
            stream: stream_options(fasta.path()),
            out: out.path(),
            config_out: None,
        })
        .unwrap();

        let collected = collected.lock().unwrap();
        assert_eq!(collected.rows.len(), streamed.precursors);
        assert_eq!(collected.rows[0].0, "PEPTIDEMR");
        assert_eq!(collected.rows[0].1, 2);
        assert!(collected.rows[0].2 > 0, "a spectrum with no peaks");
        assert_eq!(
            streamed.fragments,
            collected.rows.iter().map(|row| row.2).sum::<usize>()
        );
        // The sink is told the stream ended, exactly once, or a buffering consumer never commits
        // its last batch.
        assert_eq!(collected.finished, 1);
        assert_eq!(collected.model.as_deref(), Some("builtin:small-v0"));
        // No file, so nothing to say about one.
        assert_eq!(collected.output_present, Some(false));

        assert_eq!(streamed.precursors, written.precursors);
        assert_eq!(streamed.fragments, written.fragments);
        assert_eq!(streamed.peptides, written.peptides);
        let text = std::fs::read_to_string(out.path()).unwrap();
        assert!(text.starts_with("ModifiedPeptide\t"), "{text:.80}");
        assert_eq!(text.lines().count(), 1 + written.fragments);
    }

    /// A run that fails validation must not touch an existing library. `write_library` opens the
    /// output before it can know the settings are usable, so the file is created by a thunk that
    /// only runs once everything else has passed.
    #[test]
    fn a_rejected_run_leaves_the_existing_output_alone() {
        let fasta = tiny_fasta();
        let out = Scratch::new("precious.tsv");
        std::fs::write(out.path(), "EXISTING LIBRARY\n").unwrap();

        let mut opts = stream_options(fasta.path());
        opts.min_intensity = 5.0;
        let error = write_library(&LibraryOptions {
            stream: opts,
            out: out.path(),
            config_out: None,
        })
        .unwrap_err();

        assert!(error.to_string().contains("min_intensity"), "{error}");
        assert_eq!(
            std::fs::read_to_string(out.path()).unwrap(),
            "EXISTING LIBRARY\n"
        );
    }

    #[test]
    fn pseudo_reverse_moves_residue_modifications_with_their_residue() {
        let peptide = Peptide::new(
            "PEPTIDEK".into(),
            vec![(
                Site::Residue(2),
                msspeculator_core::proforma::parse_descriptor("UNIMOD:35").unwrap(),
            )],
        );
        let decoy = pseudo_reverse_peptide(&peptide);
        assert_eq!(decoy.sequence, "PEDITPEK");
        assert_eq!(decoy.mods[0].0, Site::Residue(5));
        assert_eq!(decoy.modified_sequence(), "PEDITP[UNIMOD:35]EK");
    }

    #[test]
    fn modifications_land_on_the_sites_they_target() {
        let fixed = vec![parse_modification_rule("C[UNIMOD:4]").unwrap()];
        let variable = vec![parse_modification_rule("M[UNIMOD:35]").unwrap()];
        let forms = modified_forms("ACDM", &fixed, &variable, 1).unwrap();
        assert_eq!(forms[1].modified_sequence(), "AC[UNIMOD:4]DM[UNIMOD:35]");
        assert_eq!(
            crate::diann::modified_peptide(&forms[1]),
            "AC(UniMod:4)DM(UniMod:35)"
        );
        assert_eq!(forms[1].mods[1].0, Site::Residue(3));
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
        assert!(forms.iter().any(
            |peptide| peptide.modified_sequence() == "AC[UNIMOD:2057]S[UNIMOD:21]M[UNIMOD:35]"
        ));
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
                output_spelling(Path::new(path)).0,
                LibraryFormat::MzSpecLib,
                "{path}"
            );
        }
        for path in ["lib.tsv", "lib.tsv.gz", "lib", "mzspeclib.tsv"] {
            assert_eq!(
                output_spelling(Path::new(path)).0,
                LibraryFormat::DiannTsv,
                "{path}"
            );
        }
    }

    /// Format and compression are one answer about one suffix. They used to be two, and disagreed
    /// on `.gz`, where `Path::extension` reads a leading dot as the stem and reports none.
    #[test]
    fn compression_is_decided_by_the_same_read_as_the_format() {
        for path in ["lib.tsv.gz", "lib.mzspeclib.txt.gz", ".gz"] {
            assert!(output_spelling(Path::new(path)).1, "{path}");
        }
        for path in ["lib.tsv", "lib.mzspeclib.txt", "lib.gzip"] {
            assert!(!output_spelling(Path::new(path)).1, "{path}");
        }
    }

    #[test]
    fn ccs_conversion_matches_alphabase_formula() {
        let mobility = ccs_to_bruker_mobility(400.0, 2, 500.0);
        assert!((mobility - 0.985056867).abs() < 1e-8, "{mobility}");
    }
}
