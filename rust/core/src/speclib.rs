//! Reader for published spectral libraries (Spectronaut and DIA-NN TSV).
//!
//! Libraries are the second source of real spectra beside the prepared PROSPECT corpus, and they
//! arrive with less than PROSPECT does: mass shifts instead of modification identities, a
//! gradient-normalised retention time instead of iRT, and usually no collision energy at all.
//! This module resolves what can be resolved against declared inputs and reports the rest rather
//! than guessing at it.
//!
//! It lives in the core crate rather than in the Python bindings because fitting a context row
//! against a library is planned for the Rust runtime, and a library parsed by two different
//! implementations is a contract that will drift.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};

use crate::chem::ION_TYPES;
use crate::peptide::{ModSpec, Peptide, Site};
use crate::proforma::unimod_spec;

/// A file mass is matched to a declared modification within this tolerance.
///
/// Deliberately tight. Libraries write mass shifts at whatever precision the search engine chose,
/// and a loose tolerance is how a phospho-like mass silently becomes something else. A shift
/// rounded past this (DIA-NN writes CysPAT as `+221.082`, which is 3.05e-4 from the table's
/// 221.081695) must be declared with its observed mass instead of widening this for everyone.
pub const AUTO_TOLERANCE: f64 = 1e-4;

/// How far a declared observed mass may sit from its accession's true delta.
///
/// Guards the escape hatch above: stating an observed mass explains rounding, it does not license
/// pointing an accession at an unrelated mass.
pub const DECLARED_TOLERANCE: f64 = 0.01;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeclibFormat {
    /// `ModifiedPeptide`, `Tr_recalibrated`, `FragmentNumber`, `Decoy`.
    SpectronautTsv,
    /// `FullUniModPeptideName`, `FragmentSeriesNumber`, `decoy`, `ExcludeFromAssay`.
    DiannTsv,
}

impl SpeclibFormat {
    /// Identify a format from the header line alone.
    ///
    /// Ordered specific to generic: the DIA-NN export is a superset of the Spectronaut columns,
    /// so its distinguishing columns are tested first or every DIA-NN file would read as
    /// Spectronaut and lose its run identity.
    pub fn detect(header: &str) -> Result<Self> {
        let columns: Vec<&str> = header.trim_end_matches(['\r', '\n']).split('\t').collect();
        let has = |name: &str| columns.contains(&name);
        if has("FragmentSeriesNumber") && has("transition_group_id") {
            return Ok(Self::DiannTsv);
        }
        if has("FragmentNumber") && has("ModifiedPeptide") {
            return Ok(Self::SpectronautTsv);
        }
        bail!(
            "unrecognised spectral library header; expected either a DIA-NN export \
             (FragmentSeriesNumber, transition_group_id) or a Spectronaut export \
             (FragmentNumber, ModifiedPeptide), found columns: {}",
            columns.join(", ")
        )
    }

    fn columns(self) -> Columns {
        match self {
            Self::SpectronautTsv => Columns {
                peptide: "ModifiedPeptide",
                charge: "PrecursorCharge",
                normalized_rt: "Tr_recalibrated",
                ion_mobility: "IonMobility",
                intensity: "RelativeIntensity",
                fragment_kind: "FragmentType",
                fragment_ordinal: "FragmentNumber",
                fragment_charge: "FragmentCharge",
                fragment_loss: "FragmentLossType",
                decoy: "Decoy",
                exclude: None,
            },
            Self::DiannTsv => Columns {
                peptide: "ModifiedPeptide",
                charge: "PrecursorCharge",
                normalized_rt: "Tr_recalibrated",
                ion_mobility: "IonMobility",
                intensity: "LibraryIntensity",
                fragment_kind: "FragmentType",
                fragment_ordinal: "FragmentSeriesNumber",
                fragment_charge: "FragmentCharge",
                fragment_loss: "FragmentLossType",
                decoy: "decoy",
                exclude: Some("ExcludeFromAssay"),
            },
        }
    }
}

struct Columns {
    peptide: &'static str,
    charge: &'static str,
    normalized_rt: &'static str,
    ion_mobility: &'static str,
    intensity: &'static str,
    fragment_kind: &'static str,
    fragment_ordinal: &'static str,
    fragment_charge: &'static str,
    fragment_loss: &'static str,
    decoy: &'static str,
    exclude: Option<&'static str>,
}

/// A modification the caller declares is present, so a bare mass can become an identity.
#[derive(Debug, Clone)]
pub struct ModAlias {
    pub accession: u32,
    /// The mass as the file spells it. `None` requires the file to agree with the UNIMOD table
    /// within [`AUTO_TOLERANCE`]; `Some` accepts a rounded spelling that would otherwise fail.
    pub observed_mass: Option<f64>,
}

/// Which retention time a library offers, and therefore what it can supervise.
#[derive(Debug, Clone)]
pub enum RetentionSource {
    /// A real retention time in minutes, in the named column.
    Column(String),
    /// `Tr_recalibrated`: a gradient fraction. It can anchor a chromatography row, but it is not
    /// an iRT and must not be supervised as one.
    Normalized,
}

#[derive(Debug, Clone)]
pub struct LibrarySpec {
    /// Names the chromatography and acquisition rows this library trains, e.g. "Evosep60SPD_hh".
    pub context: String,
    pub instrument: String,
    pub detector: String,
    pub fragmentation: String,
    pub aliases: Vec<ModAlias>,
    pub retention: RetentionSource,
    /// Honour DIA-NN's `ExcludeFromAssay`. Off by default, and deliberately so: that flag marks
    /// transitions its quantification skipped, not transitions that are wrong. Acting on it takes
    /// a measured library from a median of eleven fragments per precursor down to three.
    pub drop_excluded: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fragment {
    /// Row in the `(L-1, ION_TYPES)` grid: the site after residue `site + 1`.
    pub site: u16,
    /// Column in [`ION_TYPES`].
    pub ion: u8,
    pub intensity: f32,
}

#[derive(Debug, Clone)]
pub struct LibraryPrecursor {
    pub peptide: Peptide,
    pub charge: u8,
    pub retention: f32,
    pub ion_mobility: Option<f32>,
    pub fragments: Vec<Fragment>,
}

/// What the read discarded. Returned rather than logged: whether an unmapped mass is fatal is the
/// caller's policy, and a reader that decided silently would hide a corpus-wide mistake.
#[derive(Debug, Clone, Default)]
pub struct LibraryStats {
    pub rows: u64,
    pub decoys: u64,
    /// Rows flagged `ExcludeFromAssay`. Counted whether or not they were acted on, so the caller
    /// can see how much of the library that flag covers before deciding to honour it.
    pub excluded: u64,
    pub precursors: u64,
    /// Precursors whose every fragment fell outside the grid, and which are therefore not emitted:
    /// an empty spectrum has no defined angle against anything.
    pub precursors_without_fragments: u64,
    /// Fragments outside the grid: charge above 2, a neutral loss, or an out-of-range ordinal.
    pub fragments_dropped: u64,
    /// Mass shifts no declared alias explains, spelled as they appeared.
    pub unmapped_masses: Vec<String>,
}

/// Map one annotated fragment onto the `(site, ion)` grid, or `None` if it has no place in it.
///
/// The grid holds b and y at charges 1 and 2 with no losses, so everything else is dropped,
/// about 2.5% of a typical timsTOF library. Indexing follows `chem::fragment_mz_matrix`: row `i`
/// is the site after residue `i + 1`, so b ordinal `n` is row `n - 1` and y ordinal `n` is row
/// `length - 1 - n`. Getting this wrong transposes b against y and still produces a plausible
/// spectral angle, which is why it is derived from that one contract rather than restated.
pub fn fragment_cell(kind: char, ordinal: usize, charge: u8, length: usize) -> Option<(u16, u8)> {
    if length < 2 || ordinal < 1 || ordinal > length - 1 {
        return None;
    }
    let is_b = match kind {
        'b' | 'B' => true,
        'y' | 'Y' => false,
        _ => return None,
    };
    let ion = ION_TYPES
        .iter()
        .position(|&(b, z)| b == is_b && z == charge)? as u8;
    let site = if is_b {
        ordinal - 1
    } else {
        length - 1 - ordinal
    };
    Some((site as u16, ion))
}

/// Resolve one mass shift against the declared aliases.
fn resolve_mass(shift: f64, aliases: &[ModAlias]) -> Result<Option<ModSpec>> {
    for alias in aliases {
        let table = unimod_spec(alias.accession)?.delta_mass()?;
        let target = match alias.observed_mass {
            Some(observed) => {
                if (observed - table).abs() > DECLARED_TOLERANCE {
                    bail!(
                        "declared mass {observed} for UNIMOD:{} is {:.4} Da from its true delta \
                         {table:.6}; stating an observed mass explains a rounded spelling, not a \
                         different modification",
                        alias.accession,
                        (observed - table).abs()
                    );
                }
                observed
            }
            None => table,
        };
        if (shift - target).abs() <= AUTO_TOLERANCE {
            return Ok(Some(unimod_spec(alias.accession)?));
        }
    }
    Ok(None)
}

/// Parse a library's modified sequence into a peptide, resolving every mass against the aliases.
///
/// Accepts both spellings seen in the wild: Spectronaut wraps the sequence in underscores and
/// brackets its shifts (`_PEPT[+79.96633]IDE_`), DIA-NN parenthesises them. An unresolved shift
/// is reported through `unmapped`, leaving the caller to decide whether the run should fail.
pub fn parse_library_peptide(
    text: &str,
    aliases: &[ModAlias],
    unmapped: &mut Vec<String>,
) -> Result<Peptide> {
    let text = text.trim().trim_matches('_');
    let mut sequence = String::with_capacity(text.len());
    let mut mods: Vec<(Site, ModSpec)> = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '[' | '(' => {
                let closing = if c == '[' { ']' } else { ')' };
                let mut token = String::new();
                let mut closed = false;
                for inner in chars.by_ref() {
                    if inner == closing {
                        closed = true;
                        break;
                    }
                    token.push(inner);
                }
                if !closed {
                    bail!("unterminated modification in {text:?}");
                }
                let shift: f64 = token
                    .trim()
                    .parse()
                    .with_context(|| format!("modification {token:?} in {text:?} is not a mass shift; only mass-shift libraries are read"))?;
                // A shift before any residue belongs to the N terminus; otherwise it belongs to
                // the residue just written.
                let site = if sequence.is_empty() {
                    Site::NTerm
                } else {
                    Site::Residue(sequence.chars().count() - 1)
                };
                match resolve_mass(shift, aliases)? {
                    Some(spec) => mods.push((site, spec)),
                    None => {
                        let spelled = format!("{c}{token}{closing}");
                        if !unmapped.contains(&spelled) {
                            unmapped.push(spelled);
                        }
                    }
                }
            }
            c if c.is_ascii_alphabetic() => sequence.push(c.to_ascii_uppercase()),
            c if c.is_whitespace() => {}
            other => bail!("unexpected character {other:?} in modified sequence {text:?}"),
        }
    }
    if sequence.is_empty() {
        bail!("modified sequence {text:?} has no residues");
    }
    Ok(Peptide::new(sequence, mods))
}

fn header_index(header: &[&str], name: &str) -> Result<usize> {
    header
        .iter()
        .position(|column| *column == name)
        .ok_or_else(|| anyhow!("library is missing the {name:?} column"))
}

/// Read a spectral library into one record per precursor.
pub fn read_speclib(
    path: &Path,
    spec: &LibrarySpec,
) -> Result<(Vec<LibraryPrecursor>, LibraryStats)> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading spectral library {}", path.display()))?;
    read_speclib_str(&text, spec)
}

/// Read a library already in memory. Split out so tests need no fixture files.
pub fn read_speclib_str(
    text: &str,
    spec: &LibrarySpec,
) -> Result<(Vec<LibraryPrecursor>, LibraryStats)> {
    let mut lines = text.lines();
    let header_line = lines.next().context("spectral library is empty")?;
    let format = SpeclibFormat::detect(header_line)?;
    let columns = format.columns();
    let header: Vec<&str> = header_line.trim_end_matches('\r').split('\t').collect();

    let peptide_at = header_index(&header, columns.peptide)?;
    let charge_at = header_index(&header, columns.charge)?;
    let intensity_at = header_index(&header, columns.intensity)?;
    let kind_at = header_index(&header, columns.fragment_kind)?;
    let ordinal_at = header_index(&header, columns.fragment_ordinal)?;
    let fragment_charge_at = header_index(&header, columns.fragment_charge)?;
    let loss_at = header_index(&header, columns.fragment_loss).ok();
    let decoy_at = header_index(&header, columns.decoy).ok();
    let exclude_at = columns
        .exclude
        .and_then(|name| header_index(&header, name).ok());
    let mobility_at = header_index(&header, columns.ion_mobility).ok();
    let retention_at = match &spec.retention {
        RetentionSource::Column(name) => header_index(&header, name)?,
        RetentionSource::Normalized => header_index(&header, columns.normalized_rt)?,
    };

    let mut stats = LibraryStats::default();
    let mut order: Vec<(String, u8)> = Vec::new();
    let mut precursors: HashMap<(String, u8), LibraryPrecursor> = HashMap::new();

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        stats.rows += 1;
        let fields: Vec<&str> = line.trim_end_matches('\r').split('\t').collect();
        let field = |index: usize| -> Result<&str> {
            fields
                .get(index)
                .copied()
                .ok_or_else(|| anyhow!("row {} has {} fields, too few", stats.rows, fields.len()))
        };
        if let Some(index) = decoy_at {
            let raw = field(index)?.trim();
            if !matches!(raw, "0" | "" | "False" | "false" | "FALSE") {
                stats.decoys += 1;
                continue;
            }
        }
        if let Some(index) = exclude_at {
            if matches!(field(index)?.trim(), "True" | "true" | "TRUE" | "1") {
                stats.excluded += 1;
                if spec.drop_excluded {
                    continue;
                }
            }
        }
        let sequence = field(peptide_at)?.to_string();
        let charge: u8 = field(charge_at)?
            .trim()
            .parse()
            .with_context(|| format!("row {} has an unreadable precursor charge", stats.rows))?;
        let key = (sequence.clone(), charge);
        if !precursors.contains_key(&key) {
            let peptide =
                parse_library_peptide(&sequence, &spec.aliases, &mut stats.unmapped_masses)?;
            let retention: f32 = field(retention_at)?
                .trim()
                .parse()
                .with_context(|| format!("row {} has an unreadable retention time", stats.rows))?;
            let ion_mobility = match mobility_at {
                Some(index) => field(index)?.trim().parse::<f32>().ok(),
                None => None,
            };
            order.push(key.clone());
            precursors.insert(
                key.clone(),
                LibraryPrecursor {
                    peptide,
                    charge,
                    retention,
                    ion_mobility,
                    fragments: Vec::new(),
                },
            );
        }
        let entry = precursors.get_mut(&key).expect("inserted above");
        let length = entry.peptide.length();
        let lossless = match loss_at {
            Some(index) => matches!(field(index)?.trim(), "noloss" | "" | "none"),
            None => true,
        };
        let kind = field(kind_at)?.trim().chars().next().unwrap_or('?');
        let ordinal: usize = field(ordinal_at)?.trim().parse().unwrap_or(0);
        let fragment_charge: u8 = field(fragment_charge_at)?.trim().parse().unwrap_or(0);
        let intensity: f32 = field(intensity_at)?.trim().parse().unwrap_or(0.0);
        match fragment_cell(kind, ordinal, fragment_charge, length).filter(|_| lossless) {
            Some((site, ion)) => entry.fragments.push(Fragment {
                site,
                ion,
                intensity,
            }),
            None => stats.fragments_dropped += 1,
        }
    }

    let mut out = Vec::with_capacity(order.len());
    for key in order {
        if let Some(precursor) = precursors.remove(&key) {
            if precursor.fragments.is_empty() {
                stats.precursors_without_fragments += 1;
                continue;
            }
            out.push(precursor);
        }
    }
    stats.precursors = out.len() as u64;
    Ok((out, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPECTRONAUT: &str = "ModifiedPeptide\tStrippedPeptide\tPrecursorMz\tPrecursorCharge\tTr_recalibrated\tIonMobility\tProteinID\tDecoy\tFragmentMz\tRelativeIntensity\tFragmentType\tFragmentNumber\tFragmentCharge\tFragmentLossType
_PEPTIDEK_\tPEPTIDEK\t500.1\t2\t0.25\t0.9\tP1\t0\t100.0\t0.5\tb\t2\t1\tnoloss
_PEPTIDEK_\tPEPTIDEK\t500.1\t2\t0.25\t0.9\tP1\t0\t200.0\t1.0\ty\t3\t1\tnoloss
_PEPTIDEK_\tPEPTIDEK\t500.1\t2\t0.25\t0.9\tP1\t0\t300.0\t0.2\ty\t3\t3\tnoloss
_DECOYK_\tDECOYK\t400.1\t2\t0.30\t0.8\tP2\t1\t150.0\t0.4\tb\t2\t1\tnoloss
_PEPT[+79.96633]IDEK_\tPEPTIDEK\t540.1\t2\t0.40\t0.9\tP1\t0\t120.0\t0.7\tb\t3\t1\tnoloss
";

    fn spec(aliases: Vec<ModAlias>) -> LibrarySpec {
        LibrarySpec {
            context: "test".into(),
            instrument: "timsTOF".into(),
            detector: "TOF".into(),
            fragmentation: "HCD".into(),
            aliases,
            retention: RetentionSource::Normalized,
            drop_excluded: false,
        }
    }

    #[test]
    fn detects_both_exports_from_the_header() {
        assert_eq!(
            SpeclibFormat::detect(SPECTRONAUT.lines().next().unwrap()).unwrap(),
            SpeclibFormat::SpectronautTsv
        );
        let diann = "FileName\tPrecursorMz\tTr_recalibrated\ttransition_group_id\tModifiedPeptide\tPrecursorCharge\tFragmentType\tFragmentCharge\tFragmentSeriesNumber\tFragmentLossType\tLibraryIntensity\tIonMobility\tdecoy\tExcludeFromAssay";
        assert_eq!(
            SpeclibFormat::detect(diann).unwrap(),
            SpeclibFormat::DiannTsv
        );
        assert!(SpeclibFormat::detect("a\tb\tc").is_err());
    }

    #[test]
    fn fragment_cells_follow_the_matrix_contract() {
        // chem::fragment_mz_matrix: row i is the site after residue i+1, so b_n is row n-1 and
        // y_n is row length-1-n. A length-8 peptide has 7 sites.
        assert_eq!(fragment_cell('b', 1, 1, 8), Some((0, 0)));
        assert_eq!(fragment_cell('y', 7, 1, 8), Some((0, 1)));
        assert_eq!(fragment_cell('b', 7, 2, 8), Some((6, 2)));
        assert_eq!(fragment_cell('y', 1, 2, 8), Some((6, 3)));
        // Outside the grid: charge 3, an unknown series, and ordinals past the last site.
        assert_eq!(fragment_cell('b', 2, 3, 8), None);
        assert_eq!(fragment_cell('c', 2, 1, 8), None);
        assert_eq!(fragment_cell('b', 8, 1, 8), None);
        assert_eq!(fragment_cell('y', 0, 1, 8), None);
    }

    #[test]
    fn reads_precursors_and_reports_what_it_dropped() {
        let (precursors, stats) = read_speclib_str(
            SPECTRONAUT,
            &spec(vec![ModAlias {
                accession: 21,
                observed_mass: None,
            }]),
        )
        .unwrap();
        assert_eq!(stats.rows, 5);
        assert_eq!(stats.decoys, 1);
        assert_eq!(stats.precursors, 2);
        // The charge-3 fragment has no column in the grid.
        assert_eq!(stats.fragments_dropped, 1);
        assert!(stats.unmapped_masses.is_empty());

        let plain = &precursors[0];
        assert_eq!(plain.peptide.sequence, "PEPTIDEK");
        assert!(plain.peptide.mods.is_empty());
        assert_eq!(plain.fragments.len(), 2);
        assert_eq!(plain.charge, 2);
        assert!((plain.retention - 0.25).abs() < 1e-6);

        let modified = &precursors[1];
        assert_eq!(modified.peptide.sequence, "PEPTIDEK");
        assert_eq!(modified.peptide.mods.len(), 1);
        assert!(matches!(
            modified.peptide.mods[0],
            (Site::Residue(3), ModSpec::Unimod { accession: 21, .. })
        ));
    }

    #[test]
    fn an_undeclared_mass_is_reported_rather_than_invented() {
        let (precursors, stats) = read_speclib_str(SPECTRONAUT, &spec(vec![])).unwrap();
        assert_eq!(stats.unmapped_masses, vec!["[+79.96633]".to_string()]);
        // The peptide still parses, without the modification it could not identify.
        assert!(precursors[1].peptide.mods.is_empty());
    }

    #[test]
    fn a_rounded_mass_needs_declaring_but_a_wrong_one_is_refused() {
        // DIA-NN writes 6C-CysPAT as +221.082, which is 3.05e-4 from the table's 221.081695 and
        // so past AUTO_TOLERANCE.
        let rounded = SPECTRONAUT.replace("+79.96633", "+221.082");
        let auto = spec(vec![ModAlias {
            accession: 2057,
            observed_mass: None,
        }]);
        let (_, stats) = read_speclib_str(&rounded, &auto).unwrap();
        assert_eq!(stats.unmapped_masses, vec!["[+221.082]".to_string()]);

        let declared = spec(vec![ModAlias {
            accession: 2057,
            observed_mass: Some(221.082),
        }]);
        let (precursors, stats) = read_speclib_str(&rounded, &declared).unwrap();
        assert!(stats.unmapped_masses.is_empty());
        assert_eq!(precursors[1].peptide.mods.len(), 1);

        // Declaring a mass that belongs to a different modification is refused outright.
        let wrong = spec(vec![ModAlias {
            accession: 2057,
            observed_mass: Some(79.96633),
        }]);
        let error = read_speclib_str(&rounded, &wrong).unwrap_err().to_string();
        assert!(error.contains("from its true delta"), "{error}");
    }
}
