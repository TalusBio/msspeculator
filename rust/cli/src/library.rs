use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use pepdistill_core::peptide::{ModSpec, Peptide, Site};
use pepdistill_core::{predict_peptide_charges_prepared, Artifact, MsContext, PreparedContext};

const VALID_AA: &str = "GASPVTCLINDQKEMHFRYW";
const CCS_IM_COEF: f64 = 1059.62245;
const IM_GAS_MASS: f64 = 28.0;

pub struct LibraryOptions<'a> {
    pub model: &'a str,
    pub fasta: &'a str,
    pub out: &'a str,
    pub ms_context: Option<&'a MsContext>,
    pub chrom_context: Option<&'a str>,
    pub min_intensity: f64,
    pub missed_cleavages: usize,
    pub min_length: usize,
    pub max_length: usize,
    pub min_charge: i64,
    pub max_charge: i64,
    pub max_variable_oxidation: usize,
    pub no_fixed_carbamidomethyl: bool,
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

fn oxidation_forms(sequence: &str, max_variable: usize) -> Vec<Vec<usize>> {
    let sites: Vec<usize> = sequence
        .bytes()
        .enumerate()
        .filter_map(|(i, aa)| (aa == b'M').then_some(i))
        .collect();
    let mut forms = vec![Vec::new()];
    if max_variable == 0 {
        return forms;
    }
    fn choose(
        sites: &[usize],
        start: usize,
        left: usize,
        picked: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if left == 0 {
            out.push(picked.clone());
            return;
        }
        for i in start..=sites.len().saturating_sub(left) {
            picked.push(sites[i]);
            choose(sites, i + 1, left - 1, picked, out);
            picked.pop();
        }
    }
    for n in 1..=max_variable.min(sites.len()) {
        choose(&sites, 0, n, &mut Vec::new(), &mut forms);
    }
    forms
}

fn modified_peptide(sequence: &str, oxidized: &[usize], fixed_cam: bool) -> (Peptide, String) {
    let mut mods = Vec::new();
    let mut diann = String::new();
    for (i, aa) in sequence.chars().enumerate() {
        diann.push(aa);
        if fixed_cam && aa == 'C' {
            mods.push((
                Site::Residue(i),
                ModSpec::Named("Carbamidomethyl@C".to_string()),
            ));
            diann.push_str("(UniMod:4)");
        }
        if oxidized.contains(&i) {
            mods.push((Site::Residue(i), ModSpec::Named("Oxidation@M".to_string())));
            diann.push_str("(UniMod:35)");
        }
    }
    (Peptide::new(sequence.to_string(), mods), diann)
}

fn ccs_to_bruker_mobility(ccs: f64, charge: i64, precursor_mz: f64) -> f64 {
    let mass = precursor_mz * charge as f64;
    let reduced_mass = mass * IM_GAS_MASS / (mass + IM_GAS_MASS);
    ccs * reduced_mass.sqrt() / charge as f64 / CCS_IM_COEF
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

    let artifact = Artifact::load(opts.model)?;
    let context = PreparedContext::new(&artifact, opts.ms_context, opts.chrom_context)?;
    let charges = (opts.min_charge..=opts.max_charge).collect::<Vec<_>>();
    let mut writer = BufWriter::new(
        File::create(opts.out).with_context(|| format!("creating library {}", opts.out))?,
    );
    writeln!(writer, "ModifiedPeptide\tStrippedPeptide\tPrecursorMz\tPrecursorCharge\tTr_recalibrated\tIonMobility\tProteinID\tDecoy\tFragmentMz\tFragmentType\tFragmentNumber\tFragmentCharge\tFragmentLossType\tRelativeIntensity")?;

    let mut stats = LibraryStats {
        proteins: records.len(),
        peptides: peptides.len(),
        ..LibraryStats::default()
    };
    for (sequence, proteins) in peptides {
        let protein_group = proteins.into_iter().collect::<Vec<_>>().join(";");
        for oxidized in oxidation_forms(&sequence, opts.max_variable_oxidation) {
            let (peptide, diann_sequence) =
                modified_peptide(&sequence, &oxidized, !opts.no_fixed_carbamidomethyl);
            let predictions = predict_peptide_charges_prepared(
                &artifact,
                &peptide,
                &charges,
                &context,
                opts.min_intensity,
            )?;
            for prediction in predictions {
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
                let mobility =
                    ccs_to_bruker_mobility(prediction.ccs as f64, charge, prediction.precursor_mz);
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
                        sequence,
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
            }
        }
    }
    writer.flush()?;
    Ok(stats)
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
        let (peptide, diann) = modified_peptide("ACDM", &[3], true);
        assert_eq!(
            peptide.modified_sequence(),
            "AC[Carbamidomethyl@C]DM[Oxidation@M]"
        );
        assert_eq!(diann, "AC(UniMod:4)DM(UniMod:35)");
        assert_eq!(peptide.mods[1].0, Site::Residue(3));
    }

    #[test]
    fn ccs_conversion_matches_alphabase_formula() {
        let mobility = ccs_to_bruker_mobility(400.0, 2, 500.0);
        assert!((mobility - 0.985056867).abs() < 1e-8, "{mobility}");
    }
}
