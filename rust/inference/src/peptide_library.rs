//! Predict a library from charged, modified peptide rows without digestion or modification rules.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use msspeculator_core::peptide::Peptide;
use msspeculator_core::{predict_peptide_batch_charges_prepared, ModelSource, PreparedContext};

use crate::diann::DiannSink;
use crate::library::{
    make_spectrum_row, output_spelling, LibraryFormat, LibrarySink, LibraryStats, SpectrumIdentity,
};
use crate::mzspeclib::{check_representable, MzSpecLibSink};
use crate::progress::{Phase, ProgressFn, Reporter};
use crate::proteome::{ProteinGroup, Residues};
use crate::provenance::{resolve_peptide_provenance, write_sidecar, LibraryProvenance, Output};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoyMethod {
    PseudoReverse,
    Shuffle,
}

impl DecoyMethod {
    pub fn name(self) -> &'static str {
        match self {
            Self::PseudoReverse => "pseudo-reverse",
            Self::Shuffle => "shuffle",
        }
    }
}

pub struct PeptideLibraryOptions<'a> {
    pub model: ModelSource,
    pub peptides: &'a Path,
    pub out: &'a Path,
    pub config_out: Option<&'a Path>,
    pub min_intensity: f64,
    pub max_fragments: Option<usize>,
    pub generate_decoys: bool,
    pub decoy_method: DecoyMethod,
    pub decoy_seed: u64,
    pub progress: Option<&'a ProgressFn<'a>>,
    pub before_writing: Option<&'a dyn Fn(&LibraryProvenance)>,
}

struct Row {
    peptide: Peptide,
    charge: i64,
    proteins: Vec<String>,
    members: Vec<u32>,
    decoy: bool,
    group: String,
    generated: bool,
}

fn read_rows(path: &Path) -> Result<Vec<Row>> {
    let file = File::open(path).with_context(|| format!("opening peptides {}", path.display()))?;
    let mut lines = BufReader::new(file).lines();
    let header = lines.next().transpose()?.context("peptide TSV is empty")?;
    let columns: Vec<&str> = header.trim_end_matches('\r').split('\t').collect();
    let column = |name: &str| columns.iter().position(|value| *value == name);
    let proforma_col = column("proforma").context("peptide TSV needs proforma column")?;
    let proteins_col = column("protein_ids").context("peptide TSV needs protein_ids column")?;
    let decoy_col = column("decoy");
    let group_col = column("decoy_group");
    let mut rows = Vec::new();
    for (offset, line) in lines.enumerate() {
        let line_no = offset + 2;
        let line = line.with_context(|| format!("reading peptide TSV line {line_no}"))?;
        let fields: Vec<&str> = line.trim_end_matches('\r').split('\t').collect();
        if fields.len() != columns.len() {
            bail!(
                "peptide TSV line {line_no}: expected {} columns, got {}",
                columns.len(),
                fields.len()
            );
        }
        let notation = fields[proforma_col];
        let (sequence, charge) = notation.rsplit_once('/').with_context(|| {
            format!("peptide TSV line {line_no}: proforma must include charge, e.g. PEPTIDEK/2")
        })?;
        let charge: i64 = charge.parse().with_context(|| {
            format!("peptide TSV line {line_no}: invalid charge in {notation:?}")
        })?;
        if charge < 1 {
            bail!("peptide TSV line {line_no}: charge must be positive");
        }
        let peptide = Peptide::parse(sequence).with_context(|| {
            format!("peptide TSV line {line_no}: invalid proforma {notation:?}")
        })?;
        peptide
            .validate_mod_specs()
            .with_context(|| format!("peptide TSV line {line_no}: unsupported modification"))?;
        let proteins: Vec<String> = fields[proteins_col]
            .split(';')
            .map(str::trim)
            .map(str::to_owned)
            .collect();
        if proteins.iter().any(String::is_empty) {
            bail!("peptide TSV line {line_no}: protein_ids must contain nonempty IDs separated by semicolons");
        }
        let decoy = match decoy_col.map(|i| fields[i].trim()).unwrap_or("") {
            "" | "0" | "false" => false,
            "1" | "true" => true,
            value => {
                bail!("peptide TSV line {line_no}: invalid decoy {value:?}; use true or false")
            }
        };
        let declared_group = group_col
            .map(|i| fields[i].trim())
            .filter(|value| !value.is_empty());
        if decoy && declared_group.is_none() {
            bail!("peptide TSV line {line_no}: a supplied decoy needs decoy_group");
        }
        let group = declared_group
            .map(str::to_owned)
            .unwrap_or_else(|| notation.to_owned());
        let members = (0..proteins.len()).map(|i| i as u32).collect();
        rows.push(Row {
            peptide,
            charge,
            proteins,
            members,
            decoy,
            group,
            generated: false,
        });
    }
    if rows.is_empty() {
        bail!("peptide TSV has no rows");
    }
    validate_groups(&rows)?;
    Ok(rows)
}

fn validate_groups(rows: &[Row]) -> Result<()> {
    let mut groups: HashMap<&str, (usize, usize, i64)> = HashMap::new();
    let mut identities = HashSet::new();
    for row in rows {
        let identity = (row.peptide.modified_sequence(), row.charge);
        if !identities.insert(identity) {
            bail!(
                "duplicate charged peptide in input: {}/{}",
                row.peptide.modified_sequence(),
                row.charge
            );
        }
        let entry = groups.entry(&row.group).or_insert((0, 0, row.charge));
        if entry.2 != row.charge {
            bail!("decoy_group {:?} mixes charge states", row.group);
        }
        if row.decoy {
            entry.1 += 1;
        } else {
            entry.0 += 1;
        }
    }
    for (group, (targets, decoys, _)) in groups {
        if targets != 1 || decoys > 1 {
            bail!("decoy_group {group:?} needs one target and at most one decoy; got {targets} targets and {decoys} decoys");
        }
    }
    Ok(())
}

fn seed_for(seed: u64, sequence: &str, charge: i64, attempt: u64) -> u64 {
    let mut state = seed ^ 0xcbf29ce484222325 ^ (charge as u64) ^ attempt;
    for byte in sequence.bytes() {
        state = (state ^ byte as u64).wrapping_mul(0x100000001b3);
    }
    state
}

fn shuffle_peptide(peptide: &Peptide, seed: u64) -> Peptide {
    // Shuffle interior residues and carry residue modifications with their residues.
    let length = peptide.sequence.len();
    if length < 4 {
        return peptide.clone();
    }
    let mut order: Vec<usize> = (0..length).collect();
    let mut state = seed;
    for i in (2..length - 1).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = 1 + (state as usize % i);
        order.swap(i, j);
    }
    reorder_peptide(peptide, &order)
}

fn reverse_interior(peptide: &Peptide, flank: usize) -> Peptide {
    let length = peptide.sequence.len();
    let mut order: Vec<usize> = (0..length).collect();
    order[flank..length - flank].reverse();
    reorder_peptide(peptide, &order)
}

fn reorder_peptide(peptide: &Peptide, order: &[usize]) -> Peptide {
    let length = peptide.sequence.len();
    let bytes = peptide.sequence.as_bytes();
    let sequence: String = order.iter().map(|&index| bytes[index] as char).collect();
    let mut inverse = vec![0; length];
    for (new, &old) in order.iter().enumerate() {
        inverse[old] = new;
    }
    let mods = peptide
        .mods
        .iter()
        .map(|(site, spec)| {
            let moved = match site {
                msspeculator_core::peptide::Site::Residue(index) => {
                    msspeculator_core::peptide::Site::Residue(inverse[*index])
                }
                other => *other,
            };
            (moved, spec.clone())
        })
        .collect();
    Peptide::new(sequence, mods)
}

fn add_decoys(rows: &mut Vec<Row>, method: DecoyMethod, seed: u64) -> usize {
    // A decoy must not have the identity of any target, even when that target
    // was requested at another charge. Decoys remain charge-specific: the
    // same generated peptidoform can serve a target's /2 and /3 entries.
    let target_peptidoforms: HashSet<String> = rows
        .iter()
        .filter(|row| !row.decoy)
        .map(|row| row.peptide.modified_sequence())
        .collect();
    let mut occupied: HashSet<(String, i64)> = rows
        .iter()
        .map(|row| (row.peptide.modified_sequence(), row.charge))
        .collect();
    let supplied_groups: HashSet<String> = rows
        .iter()
        .filter(|row| row.decoy)
        .map(|row| row.group.clone())
        .collect();
    let mut generated = Vec::new();
    let mut skipped = 0;
    for row in rows
        .iter()
        .filter(|row| !row.decoy && !supplied_groups.contains(&row.group))
    {
        let mut found = false;
        let attempts = match method {
            DecoyMethod::PseudoReverse => row.peptide.sequence.len().saturating_sub(2) / 2,
            DecoyMethod::Shuffle => 32,
        };
        for attempt in 0..attempts {
            let candidate = match method {
                DecoyMethod::PseudoReverse => reverse_interior(&row.peptide, attempt + 1),
                DecoyMethod::Shuffle => shuffle_peptide(
                    &row.peptide,
                    seed_for(seed, &row.peptide.sequence, row.charge, attempt as u64),
                ),
            };
            let identity = candidate.modified_sequence();
            if !target_peptidoforms.contains(&identity) && occupied.insert((identity, row.charge)) {
                generated.push(Row {
                    peptide: candidate,
                    charge: row.charge,
                    proteins: row.proteins.clone(),
                    members: row.members.clone(),
                    decoy: true,
                    group: row.group.clone(),
                    generated: true,
                });
                found = true;
                break;
            }
        }
        if !found {
            skipped += 1;
        }
    }
    rows.extend(generated);
    skipped
}

/// Predict exactly the supplied modified peptidoforms and charges. TSV columns are
/// `proforma`, `protein_ids`, and optionally `decoy`, `decoy_group`.
pub fn write_peptide_library(opts: &PeptideLibraryOptions<'_>) -> Result<LibraryStats> {
    if !(0.0..=1.0).contains(&opts.min_intensity) {
        bail!(
            "min_intensity must be in [0, 1], got {}",
            opts.min_intensity
        );
    }
    let reporter = Reporter::new(opts.progress);
    reporter.at(Phase::Digesting, 0, 0);
    let started = Instant::now();
    let mut rows = read_rows(opts.peptides)?;
    let unique_proteins: HashSet<&str> = rows
        .iter()
        .flat_map(|row| row.proteins.iter().map(String::as_str))
        .collect();
    let proteins = unique_proteins.len();
    let input_count = rows.len();
    if opts.generate_decoys {
        let skipped = add_decoys(&mut rows, opts.decoy_method, opts.decoy_seed);
        if skipped > 0 {
            eprintln!(
                "warning: skipped {skipped} generated decoys with no collision-free {} candidate; targets remain in the library",
                opts.decoy_method.name()
            );
        }
    }
    let read_elapsed = started.elapsed();
    reporter.at(Phase::Digesting, 1, 1);
    reporter.at(Phase::Loading, 0, 0);
    let load_started = Instant::now();
    let loaded = msspeculator_core::load_source(opts.model.clone())?;
    let context = PreparedContext::new(&loaded.artifact, None, None)?;
    let load_elapsed = load_started.elapsed();
    let (format, compressed) = output_spelling(opts.out);
    if format != LibraryFormat::MzSpecLib {
        bail!("peptide input requires an .mzspeclib or .mzspeclib.txt output (optionally .gz) to retain decoy_group");
    }
    let output = Some(Output {
        path: opts.out.display().to_string(),
        format: format.name(),
        compressed,
        counts: None,
        timing: None,
    });
    let provenance = resolve_peptide_provenance(opts, output, &loaded.artifact, &loaded.digest)?;
    if let Some(report) = opts.before_writing {
        report(&provenance);
    }
    if format == LibraryFormat::MzSpecLib {
        check_representable(&provenance)?;
    }
    let file = File::create(opts.out)
        .with_context(|| format!("creating library {}", opts.out.display()))?;
    let writer: Box<dyn Write + Send> = if compressed {
        Box::new(flate2::write::GzEncoder::new(
            file,
            flate2::Compression::default(),
        ))
    } else {
        Box::new(file)
    };
    let writer = BufWriter::new(writer);
    let mut sink: Box<dyn LibrarySink> = match format {
        LibraryFormat::DiannTsv => Box::new(DiannSink { writer }),
        LibraryFormat::MzSpecLib => Box::new(MzSpecLibSink::new(writer, opts.out)),
    };
    sink.header(&provenance)?;
    let predict_started = Instant::now();
    reporter.at(Phase::Predicting, 0, rows.len() as u64);
    let mut stats = LibraryStats {
        proteins,
        peptides: input_count,
        ..LibraryStats::default()
    };
    let mut by_charge: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (index, row) in rows.iter().enumerate() {
        by_charge.entry(row.charge).or_default().push(index);
    }
    let mut done = 0;
    for (charge, indices) in by_charge {
        for chunk in indices.chunks(64) {
            let peptides: Vec<Peptide> = chunk
                .iter()
                .map(|&index| rows[index].peptide.clone())
                .collect();
            let predictions = predict_peptide_batch_charges_prepared(
                &loaded.artifact,
                &peptides,
                &[charge],
                &context,
                opts.min_intensity,
            )?;
            for (&index, predicted) in chunk.iter().zip(predictions.iter()) {
                let row = &rows[index];
                let prediction = &predicted[0];
                let protein_group = ProteinGroup::new(&row.proteins, &row.members, row.generated);
                let spectrum = make_spectrum_row(
                    SpectrumIdentity {
                        stripped: Residues::target(&row.peptide.sequence),
                        proteins: protein_group,
                        peptide: &row.peptide,
                        decoy: row.decoy,
                        decoy_pair_id: None,
                        decoy_group: Some(&row.group),
                    },
                    prediction,
                    opts.max_fragments,
                )?;
                stats.precursors += 1;
                stats.decoys += usize::from(row.decoy);
                stats.fragments += spectrum.peaks.len();
                sink.spectrum(&spectrum)?;
                done += 1;
            }
            reporter.at(Phase::Predicting, done, rows.len() as u64);
        }
    }
    sink.finish()?;
    stats.digest = read_elapsed;
    stats.load = load_elapsed;
    stats.predict = predict_started.elapsed();
    if let Some(path) = opts.config_out {
        write_sidecar(path, &provenance, &stats)?;
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scratch::Scratch;
    use msspeculator_core::BuiltinModel;

    #[test]
    fn reads_exact_modifications_charges_and_supplied_groups() {
        let input = Scratch::holding(
            "peptides.tsv",
            "proforma\tprotein_ids\tdecoy\tdecoy_group\n\
             PEC[UNIMOD:4]TIDEK/2\tP1;P2\tfalse\tcam\n\
             PECTIDEK/3\tP1\tfalse\tplain\n\
             PEK[UNIMOD:259]TIDEK/2\tPRTC\tfalse\theavy\n\
             PECTDIEK/3\tP1\ttrue\tplain\n",
        );
        let rows = read_rows(input.path()).unwrap();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].peptide.modified_sequence(), "PEC[UNIMOD:4]TIDEK");
        assert_eq!(rows[1].charge, 3);
        assert_eq!(rows[2].peptide.modified_sequence(), "PEK[UNIMOD:259]TIDEK");
        assert!(rows[3].decoy);
        assert_eq!(rows[3].group, "plain");
    }

    #[test]
    fn generated_decoys_keep_modification_identity_and_seeded_shuffle_is_stable() {
        let input = Scratch::holding(
            "cysteines.tsv",
            "proforma\tprotein_ids\nPEC[UNIMOD:4]TIDEK/2\tP1\nPECTIDEK/2\tP1\n",
        );
        let mut rows = read_rows(input.path()).unwrap();
        assert_eq!(add_decoys(&mut rows, DecoyMethod::PseudoReverse, 42), 0);
        assert_eq!(rows.len(), 4);
        assert_ne!(
            rows[2].peptide.modified_sequence(),
            rows[3].peptide.modified_sequence()
        );
        assert!(rows[2].decoy && rows[3].decoy);
        assert_eq!(rows[2].group, rows[0].group);
        let a = shuffle_peptide(
            &rows[0].peptide,
            seed_for(42, &rows[0].peptide.sequence, 2, 0),
        );
        let b = shuffle_peptide(
            &rows[0].peptide,
            seed_for(42, &rows[0].peptide.sequence, 2, 0),
        );
        assert_eq!(a.modified_sequence(), b.modified_sequence());
    }

    #[test]
    fn pseudo_reverse_narrows_its_interior_and_carries_modifications() {
        let peptide = Peptide::parse("PEPTIDEK").unwrap();
        assert_eq!(
            reverse_interior(&peptide, 1).modified_sequence(),
            "PEDITPEK"
        );
        assert_eq!(
            reverse_interior(&peptide, 2).modified_sequence(),
            "PEDITPEK"
        );
        assert_eq!(
            reverse_interior(&peptide, 3).modified_sequence(),
            "PEPITDEK"
        );

        let modified = Peptide::parse("PEC[UNIMOD:4]TIDEK").unwrap();
        assert_eq!(
            reverse_interior(&modified, 1).modified_sequence(),
            "PEDITC[UNIMOD:4]EK"
        );
        assert_eq!(
            reverse_interior(&modified, 3).modified_sequence(),
            "PEC[UNIMOD:4]ITDEK"
        );
    }

    #[test]
    fn a_target_at_another_charge_triggers_a_shorter_reversal() {
        let input = Scratch::holding(
            "colliding-targets.tsv",
            "proforma\tprotein_ids\nPEPTIDEK/2\tP1\nPEDITPEK/3\tP2\n",
        );
        let mut rows = read_rows(input.path()).unwrap();
        assert_eq!(add_decoys(&mut rows, DecoyMethod::PseudoReverse, 7), 0);
        assert_eq!(rows.len(), 4);
        let decoy = rows
            .iter()
            .find(|row| row.decoy && row.charge == 2)
            .unwrap();
        assert_eq!(decoy.peptide.modified_sequence(), "PEPITDEK");
        assert_eq!(decoy.group, rows[0].group);
    }

    #[test]
    fn a_target_can_remain_unpaired_when_every_reversal_collides() {
        let input = Scratch::holding("palindrome.tsv", "proforma\tprotein_ids\nPEEEEEEK/2\tP1\n");
        let mut rows = read_rows(input.path()).unwrap();
        assert_eq!(add_decoys(&mut rows, DecoyMethod::PseudoReverse, 7), 1);
        assert_eq!(rows.len(), 1);
        assert!(!rows[0].decoy);
    }

    #[test]
    fn charge_states_of_one_target_each_get_a_decoy() {
        let input = Scratch::holding(
            "two-charges.tsv",
            "proforma\tprotein_ids\nPEPTIDEK/2\tP1\nPEPTIDEK/3\tP1\n",
        );
        let mut rows = read_rows(input.path()).unwrap();
        assert_eq!(add_decoys(&mut rows, DecoyMethod::PseudoReverse, 7), 0);
        assert_eq!(rows.len(), 4);
        assert_eq!(
            rows[2].peptide.modified_sequence(),
            rows[3].peptide.modified_sequence()
        );
        assert_ne!(rows[2].charge, rows[3].charge);
    }

    #[test]
    fn shuffle_retries_when_its_first_candidate_is_a_target_at_another_charge() {
        let peptide = Peptide::parse("PEPTIDEK").unwrap();
        let first = shuffle_peptide(&peptide, seed_for(7, &peptide.sequence, 2, 0));
        let input = Scratch::holding(
            "shuffle-collision.tsv",
            &format!(
                "proforma\tprotein_ids\nPEPTIDEK/2\tP1\n{}/3\tP2\n",
                first.modified_sequence()
            ),
        );
        let mut rows = read_rows(input.path()).unwrap();
        assert_eq!(add_decoys(&mut rows, DecoyMethod::Shuffle, 7), 0);
        let decoy = rows
            .iter()
            .find(|row| row.decoy && row.charge == 2)
            .unwrap();
        assert_ne!(decoy.peptide.modified_sequence(), first.modified_sequence());
    }

    #[test]
    fn writes_supplied_peptidoforms_as_a_grouped_library() {
        let input = Scratch::holding(
            "input.tsv",
            "proforma\tprotein_ids\tdecoy\tdecoy_group\n\
             PECTIDEK/2\tP1\tfalse\tpair-1\n\
             PECTDIEK/2\tP1\ttrue\tpair-1\n",
        );
        let out = Scratch::new("output.mzspeclib.txt");
        let sidecar = Scratch::new("output.config.json");
        let stats = write_peptide_library(&PeptideLibraryOptions {
            model: ModelSource::Builtin(BuiltinModel::SmallV0),
            peptides: input.path(),
            out: out.path(),
            config_out: Some(sidecar.path()),
            min_intensity: 0.01,
            max_fragments: Some(4),
            generate_decoys: true,
            decoy_method: DecoyMethod::Shuffle,
            decoy_seed: 42,
            progress: None,
            before_writing: None,
        })
        .unwrap();
        assert_eq!(stats.precursors, 2);
        assert_eq!(stats.decoys, 1);
        let text = std::fs::read_to_string(out.path()).unwrap();
        assert!(text.contains("PECTIDEK/2"));
        assert!(text.contains("PECTDIEK/2"));
        assert_eq!(text.matches("msspeculator:decoy_group").count(), 2);
        assert_eq!(text.matches("other attribute value=pair-1").count(), 2);
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(sidecar.path()).unwrap()).unwrap();
        assert_eq!(
            config["inputs"]["sequence"]["path"],
            input.path().display().to_string()
        );
        assert_eq!(config["inputs"]["sequence"]["kind"], "peptides");
        assert!(config.get("digestion").is_none());
        assert_eq!(config["decoys"]["seed"], 42);
    }
}
