//! mzSpecLib text serialization.
//!
//! Written here rather than delegated to `mzannotate`: that crate pulls in mzcore/mzsignal/mzdata
//! (81 crates) and re-derives every mass from a `rustyms` peptidoform, so a modification our
//! chemistry accepts but its ontology does not would abort a library run instead of writing it.
//! What we need is the spelling, and the spelling is a few dozen lines.
//!
//! Every term below is spelled `ACCESSION|name` with the name the PSI-MS vocabulary gives that
//! accession, because readers match on both halves: `MS:1001117|theoretical mass` is silently
//! invisible to a validator that knows the accession as `theoretical neutral mass`. The output is
//! read back by the reference Python `mzspeclib` implementation in `tests/test_rust_parity.py`,
//! which also runs its validator; we are not the authority on whether this is a valid file.

use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::Path;

use anyhow::Result;

use crate::library::{LibrarySink, SpectrumRow};
use crate::provenance::{flatten, LibraryProvenance};

/// The format version of the grammar emitted here, not of our library.
const FORMAT_VERSION: &str = "1.0";

/// Prefix on every provenance key, so a reader can tell our attributes from a converter's.
const ATTRIBUTE_PREFIX: &str = "msspeculator:";

/// Spelled as `MS:1003200|software version` in the header, so it is not repeated as one of the
/// generic provenance pairs.
const VERSION_KEY: &str = "generator.version";

/// The two halves of one project-defined attribute: a name line and a value line, tied together
/// only by the group id they share.
///
/// The accession is what [`header_attributes`] matches on, and the writer builds its full term
/// from the same constant, so there is one spelling of each and a reader cannot drift from a
/// writer. Matched on the accession alone, unlike everything we write: a reader that insisted on
/// our spelling of the name half would silently skip a file that is otherwise identical.
const NAME_ACCESSION: &str = "MS:1003275";
const VALUE_ACCESSION: &str = "MS:1003276";

/// The first line of the grammar, and the cheapest way to tell a library that could carry
/// provenance from one that cannot: a DIA-NN TSV has no header to read, and looking for one to
/// the end of a multi-gigabyte file would be the same as reading the library.
const MAGIC: &str = "<mzSpecLib>";

pub struct MzSpecLibSink<W: Write> {
    writer: W,
    name: String,
    spectra: u64,
}

impl<W: Write> MzSpecLibSink<W> {
    pub fn new(writer: W, out_path: &Path) -> Self {
        // `library.mzspeclib.txt.gz` -> `library`: the name a reader shows, with our suffixes and
        // the directory stripped off.
        let name = out_path
            .file_name()
            .map_or("library", |name| name.to_str().unwrap_or("library"))
            .trim_end_matches(".gz")
            .trim_end_matches(".txt")
            .trim_end_matches(".mzspeclib")
            .to_string();
        Self {
            writer,
            name: if name.is_empty() {
                "library".to_string()
            } else {
                name
            },
            spectra: 0,
        }
    }
}

/// The provenance pairs a header carries, in the spelling [`header_attributes`] reads back.
///
/// `generator.version` drops out because the header spells it as `MS:1003200|software version`
/// rather than as one of the generic pairs.
pub(crate) fn attributes(provenance: &LibraryProvenance) -> BTreeMap<String, String> {
    let mut pairs = flatten(&provenance.to_json());
    pairs.remove(VERSION_KEY);
    pairs
}

/// The `msspeculator:` pairs a library's header carries, keyed the way [`attributes`] spells them.
///
/// One pass, stopping at the end of the header: a library carries one `<Spectrum>` per precursor
/// and there can be tens of millions of them, so the whole read costs the header. Pairs are
/// collected by attribute group, since the name and the value are two lines that nothing but the
/// group id relates, and only `msspeculator:` keys are kept — the same syntax carries any other
/// writer's attributes, and those are not ours to interpret.
///
/// An empty map means the header carried none of ours. That is every library another tool wrote,
/// and every library of ours in a format with no header to put them in. A line that is not valid
/// UTF-8 is one of those too: a file we cannot read carries none of our attributes by definition,
/// and refusing to answer would be worse than answering "nothing here".
pub(crate) fn header_attributes(reader: impl BufRead) -> BTreeMap<String, String> {
    // `map_while` rather than a raise: an I/O failure and a non-UTF-8 line both end the read,
    // because a file we cannot read carries none of our attributes by definition.
    let mut lines = reader.lines().map_while(Result::ok);
    if lines.next().as_deref().map(str::trim_end) != Some(MAGIC) {
        return BTreeMap::new();
    }
    let mut names = BTreeMap::new();
    let mut values = BTreeMap::new();
    for line in lines {
        // Every section after the header opens with `<`, so this is where the header ends. Wider
        // than `<Spectrum` on purpose: an attribute set may one day carry a grouped pair of ours
        // — `spectrum` already writes one per decoy — and those are not provenance.
        if line.starts_with('<') {
            break;
        }
        let Some((group, term, value)) = attribute_line(&line) else {
            continue;
        };
        let accession = term.split('|').next().unwrap_or(term);
        if accession == NAME_ACCESSION {
            names.insert(group.to_string(), value.to_string());
        } else if accession == VALUE_ACCESSION {
            values.insert(group.to_string(), value.to_string());
        }
    }
    names
        .into_iter()
        .filter_map(|(group, name)| {
            let key = name.strip_prefix(ATTRIBUTE_PREFIX)?;
            Some((key.to_string(), values.remove(&group)?))
        })
        .collect()
}

/// Whether every provenance value can be written and read back unchanged.
///
/// One attribute is one line, so a value carrying a line ending is a value this grammar cannot
/// hold: written raw it splits in two, and [`header_attributes`] reads back something other than
/// what was recorded. The format has no escape for it, so the only truthful answer is to refuse.
///
/// Asked twice on purpose. [`MzSpecLibSink::header`] asks because it is the authority on what it
/// can write, and [`crate::library::write_library`] asks before it creates the file, because a
/// library must not be truncated for a run that was going to fail anyway.
pub(crate) fn check_representable(provenance: &LibraryProvenance) -> Result<()> {
    for (key, value) in attributes(provenance) {
        if value.contains(['\n', '\r']) {
            anyhow::bail!(
                "{key} contains a line ending, which an mzSpecLib attribute cannot carry: \
                 {value:?}"
            );
        }
    }
    Ok(())
}

/// `[3]MS:1003276|other attribute value=4242` -> `("3", "MS:1003276|other attribute value", "4242")`.
///
/// Split on the first `=` only: a value is free text and several of ours contain one.
fn attribute_line(line: &str) -> Option<(&str, &str, &str)> {
    let (group, rest) = line.strip_prefix('[')?.split_once(']')?;
    let (term, value) = rest.split_once('=')?;
    Some((group, term, value))
}

/// mzPAF annotation for one of our fragments, e.g. `y7` or `b2^2`.
///
/// Charge is spelled only when it is not 1, matching mzPAF's own writer; our fragments carry no
/// neutral loss, so there is nothing after the ordinal.
fn annotation(ion: &str, ordinal: i64, charge: i64) -> String {
    if charge == 1 {
        format!("{ion}{ordinal}")
    } else {
        format!("{ion}{ordinal}^{charge}")
    }
}

impl<W: Write + Send> LibrarySink for MzSpecLibSink<W> {
    fn header(&mut self, provenance: &LibraryProvenance) -> Result<()> {
        writeln!(self.writer, "{MAGIC}")?;
        writeln!(
            self.writer,
            "MS:1003186|library format version={FORMAT_VERSION}"
        )?;
        writeln!(self.writer, "MS:1003188|library name={}", self.name)?;
        writeln!(
            self.writer,
            "MS:1003200|software version={}",
            provenance.generator.version
        )?;
        // `MS:1003207|library creation software` is deliberately absent: its value has to be a
        // child term of itself, and the vocabulary's children are Spectronaut, SpectraST,
        // BiblioSpec, PeakForest, DIA-NN and CompoundDb. Claiming one of those would be a lie
        // about what wrote the file, and the rule that asks for it is a MAY.
        //
        // Nothing else spells "the model and the settings that produced this library" either, so
        // the provenance rides as name/value pairs; the grammar's own escape hatch
        // for an attribute the vocabulary has no term for. One group per key: the pair is the
        // attribute, which is why both lines carry the same group id.
        check_representable(provenance)?;
        for (group, (key, value)) in attributes(provenance).into_iter().enumerate() {
            let group = group + 1;
            writeln!(
                self.writer,
                "[{group}]{NAME_ACCESSION}|other attribute name={ATTRIBUTE_PREFIX}{key}"
            )?;
            writeln!(
                self.writer,
                "[{group}]{VALUE_ACCESSION}|other attribute value={value}"
            )?;
        }
        // A set named `all` is applied to every entry of its kind without the entry referencing
        // it, so anything constant across the library belongs here and nowhere else. At 60M
        // spectra these four lines are gigabytes of repetition.
        //
        // What this library is: nothing was measured, and nothing was aggregated from replicates.
        // Both terms take the same value, which is what the reference DIA-NN converter writes.
        writeln!(self.writer, "<AttributeSet Spectrum=all>")?;
        writeln!(self.writer, "MS:1000511|ms level=2")?;
        writeln!(
            self.writer,
            "MS:1003072|spectrum origin type=MS:1003074|predicted spectrum"
        )?;
        writeln!(
            self.writer,
            "MS:1003065|spectrum aggregation type=MS:1003074|predicted spectrum"
        )?;
        // Generated decoys are predicted spectra for pseudo-reversed, unnatural peptidoforms.
        // The attribute set keeps this one annotation out of every target entry; each decoy
        // claims it below with `MS:1003212|library attribute set name`.
        writeln!(self.writer, "<AttributeSet Spectrum=Decoy>")?;
        writeln!(
            self.writer,
            "MS:1003072|spectrum origin type=MS:1003195|unnatural peptidoform decoy spectrum"
        )?;
        // No `<AttributeSet Interpretation=all>`: the reference reader crashes on one, because
        // `_new_interpretation` applies sets with an `owner_id` its attribute manager does not
        // accept. Entries carry no interpretation at all instead; with exactly one analyte per
        // spectrum there is nothing for one to disambiguate, and the validator agrees.
        Ok(())
    }

    fn spectrum(&mut self, row: &SpectrumRow<'_>) -> Result<()> {
        self.spectra += 1;
        let index = self.spectra;
        let ion = format!("{}/{}", row.proforma, row.charge);
        writeln!(self.writer, "<Spectrum={index}>")?;
        writeln!(self.writer, "MS:1003061|library spectrum name={ion}")?;
        if row.decoy {
            writeln!(self.writer, "MS:1003212|library attribute set name=Decoy")?;
        }
        if let Some(pair_id) = row.decoy_pair_id {
            // No PSI-MS term identifies the target paired with a decoy. Keep the relationship as
            // a project-defined spectrum attribute; each spectrum has one analyte in this file.
            writeln!(
                self.writer,
                "[3]{NAME_ACCESSION}|other attribute name={ATTRIBUTE_PREFIX}decoy_pair_id"
            )?;
            writeln!(
                self.writer,
                "[3]{VALUE_ACCESSION}|other attribute value={pair_id}"
            )?;
        }
        // No `MS:1003062|library spectrum index`: a reader assigns that from position as it goes,
        // and stating our own 1-based count alongside it leaves two indices in one entry.
        writeln!(self.writer, "MS:1000041|charge state={}", row.charge)?;
        writeln!(
            self.writer,
            "MS:1000744|selected ion m/z={:.8}",
            row.precursor_mz
        )?;
        // `ms level`, origin type and aggregation type are in the header's `Spectrum=all` set.
        //
        // Retention needs a unit, and the unit is a second attribute in the same group. An index
        // is not a duration, so `minute` overstates it; but the vocabulary constrains
        // `MS:1000896` to second or minute, and both `UO:0000186|dimensionless unit` and omitting
        // the unit are MUST violations. There is no value-bearing unitless retention term to
        // switch to (`MS:1002005` names a calibration standard, `MS:4000149` is a formula), so the
        // alternative is no CV retention term at all, which hides the RT from every reader. The
        // header says what the index really is; `msspeculator:retention.normalized.kind` spells it.
        //
        // With a chromatography context both quantities exist and both are reported, in separate
        // groups: the gradient time under the term whose unit claim is simply true, and the index
        // under the normalized term. A reader wanting a duration then has a real one.
        match row.irt {
            Some(irt) => {
                writeln!(self.writer, "[1]MS:1000894|retention time={:.6}", row.rt)?;
                writeln!(self.writer, "[1]UO:0000000|unit=UO:0000031|minute")?;
                writeln!(
                    self.writer,
                    "[2]MS:1000896|normalized retention time={irt:.6}"
                )?;
                writeln!(self.writer, "[2]UO:0000000|unit=UO:0000031|minute")?;
            }
            None => {
                writeln!(
                    self.writer,
                    "[1]MS:1000896|normalized retention time={:.6}",
                    row.rt
                )?;
                writeln!(self.writer, "[1]UO:0000000|unit=UO:0000031|minute")?;
            }
        }
        writeln!(
            self.writer,
            "MS:1002815|inverse reduced ion mobility={:.8}",
            row.mobility
        )?;
        writeln!(
            self.writer,
            "MS:1003059|number of peaks={}",
            row.peaks.len()
        )?;
        writeln!(self.writer, "<Analyte=1>")?;
        writeln!(
            self.writer,
            "MS:1003270|proforma peptidoform ion notation={ion}"
        )?;
        writeln!(
            self.writer,
            "MS:1000888|stripped peptide sequence={}",
            row.stripped
        )?;
        writeln!(
            self.writer,
            "MS:1003043|number of residues={}",
            row.stripped.len()
        )?;
        // The specific child of `MS:1001117|theoretical neutral mass`, because that is what this
        // is: monoisotopic, not average. Readers match on `accession|name`, so the name has to be
        // the vocabulary's spelling and not the one the validator's own rule file still carries.
        writeln!(
            self.writer,
            "MS:1003637|theoretical neutral monoisotopic mass={:.8}",
            row.neutral_mass
        )?;
        // The protein group is one attribute per member, not one joined string: a reader that
        // wants the group can rebuild it, and one that wants an accession cannot unjoin it.
        for protein in row.proteins.iter() {
            writeln!(self.writer, "MS:1000885|protein accession={protein}")?;
        }
        writeln!(self.writer, "<Peaks>")?;
        // Peak lists are m/z ordered here; our fragments come out in (position, ion type) order.
        // Ties broken by original index so a regenerated library is byte-identical.
        let mut order: Vec<usize> = (0..row.peaks.len()).collect();
        order.sort_by(|a, b| {
            row.peaks[*a]
                .mz
                .partial_cmp(&row.peaks[*b].mz)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(b))
        });
        for i in order {
            let peak = &row.peaks[i];
            writeln!(
                self.writer,
                "{:.6}\t{:.6}\t{}",
                peak.mz,
                peak.intensity,
                annotation(peak.ion, peak.ordinal, peak.charge)
            )?;
        }
        writeln!(self.writer)?;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::library::{Peak, SpectrumRow};
    use crate::proteome::{ProteinGroup, Residues};

    fn row_with_irt(irt: Option<f32>) -> SpectrumRow<'static> {
        SpectrumRow { irt, ..row() }
    }

    /// The same shape the digest hands out: an identifier table plus this peptide's indices into
    /// it, so the writer tests run the arrangement production runs.
    static IDENTIFIERS: std::sync::LazyLock<Vec<String>> =
        std::sync::LazyLock::new(|| vec!["P1".to_string(), "P2".to_string()]);
    static MEMBERS: [u32; 2] = [0, 1];
    static PEPTIDE: std::sync::LazyLock<msspeculator_core::peptide::Peptide> =
        std::sync::LazyLock::new(|| {
            msspeculator_core::peptide::Peptide::new("PEPTIDEK".to_string(), Vec::new())
        });

    fn row() -> SpectrumRow<'static> {
        SpectrumRow {
            stripped: Residues::target("PEPTIDEK"),
            proteins: ProteinGroup::new(&IDENTIFIERS, &MEMBERS, false),
            peptide: &PEPTIDE,
            proforma: "PEPTIDEK",
            decoy: false,
            decoy_pair_id: None,
            charge: 2,
            precursor_mz: 456.75,
            neutral_mass: 911.48544707,
            rt: 31.5,
            irt: None,
            mobility: 0.98,
            peaks: vec![
                Peak {
                    mz: 500.25,
                    intensity: 1.0,
                    ion: "y",
                    ordinal: 4,
                    charge: 1,
                },
                Peak {
                    mz: 200.1,
                    intensity: 0.5,
                    ion: "b",
                    ordinal: 2,
                    charge: 2,
                },
            ],
        }
    }

    /// A provenance with no acquisition context and no output, so the header's handling of the
    /// optional halves is exercised alongside the populated ones.
    fn provenance() -> LibraryProvenance {
        use crate::provenance::*;
        LibraryProvenance {
            generator: Generator {
                tool: "msspeculator-cli library",
                version: "0.1.0",
                commit: "abc123",
            },
            settings: Settings {
                inputs: Inputs {
                    model: "m.safetensors".into(),
                    model_blake2b_256: "0".repeat(64),
                    activation_override: None,
                    fasta: "proteome.fasta".into(),
                    fasta_blake2b_256: "1".repeat(64),
                },
                digestion: Digestion {
                    enzyme: "trypsin",
                    missed_cleavages: 2,
                    min_length: 7,
                    max_length: 30,
                    min_charge: 2,
                    max_charge: 4,
                },
                modifications: Modifications {
                    fixed: Vec::new(),
                    variable: vec!["M[UNIMOD:35]".into()],
                    max_variable_mods: 1,
                },
                context: Contexts {
                    ms: None,
                    chrom: None,
                },
                fragments: FragmentPolicy {
                    min_intensity: 0.01,
                    max_fragments: Some(15),
                },
                decoys: DecoyPolicy {
                    enabled: false,
                    method: "pseudo-reverse",
                    protein_prefix: "DECOY_",
                    collision_policy: "skip",
                },
            },
            retention: Retention {
                normalized: NormalizedRetention {
                    term: "MS:1000896|normalized retention time",
                    kind: "dimensionless index, minutes-like",
                    scale: "anchored",
                    anchor_check: AnchorCheck {
                        on_scale: true,
                        max_abs_error: 0.5,
                        anchors: Vec::new(),
                    },
                },
                raw: None,
            },
            output: None,
        }
    }

    /// `irt` present means a chromatography context was applied, so `rt` is a gradient time.
    fn rendered(irt: Option<f32>) -> String {
        let mut sink = MzSpecLibSink::new(Vec::new(), Path::new("out/lib.mzspeclib.txt.gz"));
        sink.header(&provenance()).unwrap();
        sink.spectrum(&row_with_irt(irt)).unwrap();
        String::from_utf8(sink.writer).unwrap()
    }

    #[test]
    fn header_carries_the_resolved_configuration() {
        let text = rendered(None);
        assert!(text.starts_with("<mzSpecLib>\nMS:1003186|library format version=1.0\n"));
        assert!(text.contains("MS:1003188|library name=lib\n"));
        assert!(text.contains("msspeculator:generator.commit"));
        assert!(text.contains("other attribute value=abc123"));
        assert!(text.contains("MS:1003275|other attribute name=msspeculator:inputs.model"));
        assert!(text.contains("MS:1003276|other attribute value=m.safetensors"));
        assert!(text.contains("MS:1003276|other attribute value=15"));
        assert!(text.contains(r#"MS:1003276|other attribute value=["M[UNIMOD:35]"]"#));
        // A knob that was not set says nothing a reader can act on: no acquisition context was
        // requested, and no file was written.
        assert!(!text.contains("context.ms"));
        assert!(!text.contains("output."));
    }

    #[test]
    fn every_attribute_written_reads_back() {
        let text = rendered(None);
        assert_eq!(
            header_attributes(text.as_bytes()),
            attributes(&provenance()),
            "the header is the only copy of the provenance a reader gets"
        );
    }

    #[test]
    fn reading_stops_at_the_end_of_the_header() {
        let mut sink = MzSpecLibSink::new(Vec::new(), Path::new("lib.mzspeclib.txt"));
        sink.header(&provenance()).unwrap();
        sink.spectrum(&SpectrumRow {
            decoy_pair_id: Some(4242),
            ..row()
        })
        .unwrap();
        let text = String::from_utf8(sink.writer).unwrap();
        // Written with the same group syntax as the header's pairs, one per spectrum, and not
        // provenance. Reading it back would be reading the library.
        assert!(text.contains("decoy_pair_id"));
        assert!(!header_attributes(text.as_bytes()).contains_key("decoy_pair_id"));
    }

    #[test]
    fn nothing_we_cannot_read_carries_attributes() {
        for bytes in [
            &b"PrecursorMz\tProductMz\n100.0\t200.0\n"[..],
            // Not UTF-8, so not a header we wrote — and not an error either.
            &[0xffu8, 0xfe, 0x00, 0x41, 0x0a, 0xc3][..],
            &b""[..],
        ] {
            assert!(header_attributes(bytes).is_empty(), "{bytes:?}");
        }
    }

    /// One attribute is one line, so a value carrying a line ending would be written raw and read
    /// back as something else. `--ms-context` factors are free text, so a value like this can
    /// reach the writer; refusing is the only spelling that does not quietly record a lie.
    #[test]
    fn a_value_that_cannot_survive_the_round_trip_is_refused() {
        let mut provenance = provenance();
        provenance.settings.context.chrom = Some("ds\nA".into());
        let mut sink = MzSpecLibSink::new(Vec::new(), Path::new("lib.mzspeclib.txt"));
        let error = sink.header(&provenance).unwrap_err().to_string();
        assert!(error.contains("context.chrom"), "{error}");
        assert!(error.contains("line ending"), "{error}");
    }

    #[test]
    fn peaks_are_mz_ordered_and_annotated_as_mzpaf() {
        let text = rendered(None);
        let peaks = text.split("<Peaks>\n").nth(1).unwrap();
        let lines: Vec<&str> = peaks.lines().filter(|line| !line.is_empty()).collect();
        assert_eq!(
            lines,
            vec!["200.100000\t0.500000\tb2^2", "500.250000\t1.000000\ty4"]
        );
    }

    #[test]
    fn an_index_only_entry_reports_only_the_normalized_term() {
        // `minute` because the vocabulary constrains this term to second or minute; the header
        // records that the number is an index. There is no gradient time to report here.
        let text = rendered(None);
        assert!(text.contains(
            "[1]MS:1000896|normalized retention time=31.500000\n\
             [1]UO:0000000|unit=UO:0000031|minute\n"
        ));
        assert!(!text.contains("MS:1000894|retention time"));
    }

    #[test]
    fn a_chromatography_context_reports_the_gradient_time_and_the_index() {
        // Two quantities, two groups: the duration under the term whose unit claim is true, the
        // index under the normalized term. A reader wanting minutes has real ones.
        let text = rendered(Some(37.75));
        assert!(text.contains(
            "[1]MS:1000894|retention time=31.500000\n\
             [1]UO:0000000|unit=UO:0000031|minute\n\
             [2]MS:1000896|normalized retention time=37.750000\n\
             [2]UO:0000000|unit=UO:0000031|minute\n"
        ));
    }

    #[test]
    fn library_constants_live_in_an_all_set_rather_than_in_every_entry() {
        let text = rendered(None);
        assert!(text.contains(
            "<AttributeSet Spectrum=all>\n\
             MS:1000511|ms level=2\n\
             MS:1003072|spectrum origin type=MS:1003074|predicted spectrum\n\
             MS:1003065|spectrum aggregation type=MS:1003074|predicted spectrum\n"
        ));
        // Each constant is stated once, in the set, and never repeated in the entry. No
        // interpretation is written at all, and no `Interpretation=all` set: one crashes the
        // reference reader, and with a single analyte there is nothing to disambiguate.
        assert!(!text.contains("AttributeSet Interpretation"));
        let entry = text.split("<Spectrum=1>").nth(1).unwrap();
        for absent in [
            "ms level",
            "spectrum origin type",
            "spectrum aggregation type",
            "<Interpretation=",
            "analyte mixture members",
        ] {
            assert!(!entry.contains(absent), "{absent} written per spectrum");
        }
    }

    #[test]
    fn protein_group_members_are_separate_attributes() {
        let text = rendered(None);
        assert!(text.contains("MS:1000885|protein accession=P1\nMS:1000885|protein accession=P2\n"));
    }

    #[test]
    fn decoy_spectra_claim_the_decoy_attribute_set() {
        let mut sink = MzSpecLibSink::new(Vec::new(), Path::new("out/lib.mzspeclib.txt"));
        sink.header(&provenance()).unwrap();
        let mut row = row();
        row.decoy = true;
        sink.spectrum(&row).unwrap();
        let text = String::from_utf8(sink.writer).unwrap();
        assert!(text.contains(
            "<AttributeSet Spectrum=Decoy>\n\
             MS:1003072|spectrum origin type=MS:1003195|unnatural peptidoform decoy spectrum\n"
        ));
        assert!(text.contains("MS:1003212|library attribute set name=Decoy\n"));
    }

    #[test]
    fn target_and_decoy_spectra_carry_the_same_pair_id() {
        let mut target = row();
        target.decoy_pair_id = Some(4242);
        let mut decoy = row();
        decoy.decoy = true;
        decoy.decoy_pair_id = Some(4242);
        let mut sink = MzSpecLibSink::new(Vec::new(), Path::new("out/lib.mzspeclib.txt"));
        sink.header(&provenance()).unwrap();
        sink.spectrum(&target).unwrap();
        sink.spectrum(&decoy).unwrap();
        let text = String::from_utf8(sink.writer).unwrap();
        assert_eq!(
            text.matches("MS:1003275|other attribute name=msspeculator:decoy_pair_id")
                .count(),
            2
        );
        assert_eq!(
            text.matches("MS:1003276|other attribute value=4242")
                .count(),
            2
        );
    }
}
