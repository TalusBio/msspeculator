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

fn write_prediction<W: Write>(
    writer: &mut W,
    stats: &mut LibraryStats,
    stripped: &str,
    protein_group: &str,
    diann_sequence: &str,
    prediction: Prediction,
) -> Result<()> {
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
    stats.precursors += 1;
    let mobility = ccs_to_bruker_mobility(prediction.ccs as f64, charge, prediction.precursor_mz);
    if !mobility.is_finite() || mobility <= 0.0 {
        bail!(
            "non-physical mobility for {} charge {}: {}",
            diann_sequence,
            charge,
            mobility
        );
    }
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
        writeln!(
            writer,
            "{}\t{}\t{:.8}\t{}\t{:.6}\t{:.8}\t{}\t0\t{:.8}\t{}\t{}\t{}\tnoloss\t{:.8}",
            diann_sequence,
            stripped,
            prediction.precursor_mz,
            charge,
            prediction.rt,
            mobility,
            protein_group,
            fragment_mz,
            prediction.fragments.ion[i],
            prediction.fragments.ord[i],
            prediction.fragments.z[i],
            intensity,
        )?;
        stats.fragments += 1;
    }
    Ok(())
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

fn write_predicted_batch<W: Write>(
    writer: &mut W,
    stats: &mut LibraryStats,
    batch: Vec<PredictedPeptide>,
) -> Result<()> {
    for item in batch {
        for prediction in item.predictions {
            write_prediction(
                writer,
                stats,
                &item.stripped,
                &item.protein_group,
                &item.diann_sequence,
                prediction,
            )?;
        }
    }
    Ok(())
}

pub fn write_diann_tsv(opts: &LibraryOptions<'_>) -> Result<LibraryStats> {
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

    let mut artifact = Artifact::load(opts.model)?;
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

    let writer_file =
        File::create(opts.out).with_context(|| format!("creating library {}", opts.out))?;
    let writer_handle = thread::spawn(move || -> Result<LibraryStats> {
        let mut writer = BufWriter::new(writer_file);
        writeln!(writer, "ModifiedPeptide\tStrippedPeptide\tPrecursorMz\tPrecursorCharge\tTr_recalibrated\tIonMobility\tProteinID\tDecoy\tFragmentMz\tFragmentType\tFragmentNumber\tFragmentCharge\tFragmentLossType\tRelativeIntensity")?;
        let mut stats = stats;
        for result in result_rx {
            write_predicted_batch(&mut writer, &mut stats, result?)?;
        }
        writer.flush()?;
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
    writer_handle
        .join()
        .map_err(|_| anyhow::anyhow!("library writer panicked"))?
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
    fn ccs_conversion_matches_alphabase_formula() {
        let mobility = ccs_to_bruker_mobility(400.0, 2, 500.0);
        assert!((mobility - 0.985056867).abs() < 1e-8, "{mobility}");
    }
}
