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
//! which also runs its validator -- we are not the authority on whether this is a valid file.

use std::io::Write;
use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use crate::library::{LibrarySink, SpectrumRow};

/// The format version of the grammar emitted here, not of our library.
const FORMAT_VERSION: &str = "1.0";

/// Prefix on every provenance key, so a reader can tell our attributes from a converter's.
const ATTRIBUTE_PREFIX: &str = "pepdistill:";

/// Spelled as `MS:1003200|software version` in the header, so it is not repeated as one of the
/// generic provenance pairs.
const VERSION_KEY: &str = "generator.version";

pub struct MzSpecLibSink<W: Write> {
    writer: W,
    name: String,
    /// True when retention is predicted iRT (no `--chrom-context`) rather than a dataset's own
    /// gradient time. The two are different quantities and get different terms.
    normalized_retention: bool,
    spectra: u64,
}

impl<W: Write> MzSpecLibSink<W> {
    pub fn new(writer: W, out_path: &str, normalized_retention: bool) -> Self {
        // `library.mzspeclib.txt.gz` -> `library`: the name a reader shows, with our suffixes and
        // the directory stripped off.
        let name = Path::new(out_path)
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
            normalized_retention,
            spectra: 0,
        }
    }
}

/// Flatten the resolved configuration into dotted `key -> value` pairs, dropping nulls.
///
/// Objects recurse; anything else is a leaf. Arrays stay JSON text: `["C[UNIMOD:2057]", ...]`
/// reads as what was passed on the command line, which is the point of recording it.
fn provenance_attributes(config: &Value) -> Vec<(String, String)> {
    fn walk(prefix: &str, value: &Value, out: &mut Vec<(String, String)>) {
        match value {
            Value::Object(fields) => {
                for (key, value) in fields {
                    let key = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{prefix}.{key}")
                    };
                    walk(&key, value, out);
                }
            }
            // A knob that was not set says nothing a reader can use, and the sidecar keeps the
            // explicit null for anyone who wants to see that it was considered.
            Value::Null => {}
            Value::String(text) => out.push((prefix.to_string(), text.clone())),
            leaf => out.push((prefix.to_string(), leaf.to_string())),
        }
    }
    let mut out = Vec::new();
    walk("", config, &mut out);
    out.retain(|(key, _)| key != VERSION_KEY);
    out
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

impl<W: Write> LibrarySink for MzSpecLibSink<W> {
    fn header(&mut self, config: &Value) -> Result<()> {
        writeln!(self.writer, "<mzSpecLib>")?;
        writeln!(
            self.writer,
            "MS:1003186|library format version={FORMAT_VERSION}"
        )?;
        writeln!(self.writer, "MS:1003188|library name={}", self.name)?;
        if let Some(version) = config
            .pointer("/generator/version")
            .and_then(serde_json::Value::as_str)
        {
            writeln!(self.writer, "MS:1003200|software version={version}")?;
        }
        // `MS:1003207|library creation software` is deliberately absent: its value has to be a
        // child term of itself, and the vocabulary's children are Spectronaut, SpectraST,
        // BiblioSpec, PeakForest, DIA-NN and CompoundDb. Claiming one of those would be a lie
        // about what wrote the file, and the rule that asks for it is a MAY.
        //
        // Nothing else spells "the model and the settings that produced this library" either, so
        // the resolved configuration rides as name/value pairs -- the grammar's own escape hatch
        // for an attribute the vocabulary has no term for. One group per key: the pair is the
        // attribute, which is why both lines carry the same group id.
        for (group, (key, value)) in provenance_attributes(config).into_iter().enumerate() {
            let group = group + 1;
            writeln!(
                self.writer,
                "[{group}]MS:1003275|other attribute name={ATTRIBUTE_PREFIX}{key}"
            )?;
            writeln!(
                self.writer,
                "[{group}]MS:1003276|other attribute value={value}"
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
        // No `<AttributeSet Interpretation=all>`: the reference reader crashes on one, because
        // `_new_interpretation` applies sets with an `owner_id` its attribute manager does not
        // accept. Entries carry no interpretation at all instead -- with exactly one analyte per
        // spectrum there is nothing for one to disambiguate, and the validator agrees.
        Ok(())
    }

    fn spectrum(&mut self, row: &SpectrumRow<'_>) -> Result<()> {
        self.spectra += 1;
        let index = self.spectra;
        let ion = format!("{}/{}", row.proforma, row.charge);
        writeln!(self.writer, "<Spectrum={index}>")?;
        writeln!(self.writer, "MS:1003061|library spectrum name={ion}")?;
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
        // Retention needs a unit, and the unit is a second attribute in the same group. An iRT is
        // an index against a normalization standard rather than a duration, so `minute` overstates
        // it -- but the vocabulary constrains `MS:1000896` to second or minute, and both
        // `UO:0000186|dimensionless unit` and omitting the unit are MUST violations. There is no
        // value-bearing unitless retention term to switch to (`MS:1002005` names a calibration
        // standard, `MS:4000149` is a formula). So the choice is this or no CV retention term at
        // all, and dropping it would hide the RT from every standard reader.
        let (retention_term, unit) = if self.normalized_retention {
            ("MS:1000896|normalized retention time", "UO:0000031|minute")
        } else {
            ("MS:1000894|retention time", "UO:0000031|minute")
        };
        writeln!(self.writer, "[1]{retention_term}={:.6}", row.rt)?;
        writeln!(self.writer, "[1]UO:0000000|unit={unit}")?;
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
        for accession in row.protein_group.split(';') {
            writeln!(self.writer, "MS:1000885|protein accession={accession}")?;
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

    fn row() -> SpectrumRow<'static> {
        SpectrumRow {
            stripped: "PEPTIDEK",
            protein_group: "P1;P2",
            diann_sequence: "PEPTIDEK",
            proforma: "PEPTIDEK",
            charge: 2,
            precursor_mz: 456.75,
            neutral_mass: 911.48544707,
            rt: 31.5,
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

    fn rendered(normalized_retention: bool) -> String {
        let mut sink =
            MzSpecLibSink::new(Vec::new(), "out/lib.mzspeclib.txt.gz", normalized_retention);
        sink.header(&serde_json::json!({
            "inputs": {"model": "m.safetensors", "fasta": null},
            "fragments": {"max_fragments": 15},
            "modifications": {"variable": ["M[UNIMOD:35]"]},
        }))
        .unwrap();
        sink.spectrum(&row()).unwrap();
        String::from_utf8(sink.writer).unwrap()
    }

    #[test]
    fn header_carries_the_resolved_configuration() {
        let text = rendered(true);
        assert!(text.starts_with("<mzSpecLib>\nMS:1003186|library format version=1.0\n"));
        assert!(text.contains("MS:1003188|library name=lib\n"));
        assert!(text.contains("MS:1003275|other attribute name=pepdistill:inputs.model"));
        assert!(text.contains("MS:1003276|other attribute value=m.safetensors"));
        assert!(text.contains("MS:1003276|other attribute value=15"));
        assert!(text.contains(r#"MS:1003276|other attribute value=["M[UNIMOD:35]"]"#));
        // A null knob says nothing a reader can act on.
        assert!(!text.contains("inputs.fasta"));
    }

    #[test]
    fn peaks_are_mz_ordered_and_annotated_as_mzpaf() {
        let text = rendered(true);
        let peaks = text.split("<Peaks>\n").nth(1).unwrap();
        let lines: Vec<&str> = peaks.lines().filter(|line| !line.is_empty()).collect();
        assert_eq!(
            lines,
            vec!["200.100000\t0.500000\tb2^2", "500.250000\t1.000000\ty4"]
        );
    }

    #[test]
    fn retention_term_distinguishes_irt_from_a_dataset_gradient() {
        // Both carry `minute` because the vocabulary constrains these terms to second or minute;
        // what differs is which quantity is being reported.
        assert!(rendered(true).contains(
            "[1]MS:1000896|normalized retention time=31.500000\n\
             [1]UO:0000000|unit=UO:0000031|minute\n"
        ));
        assert!(rendered(false).contains(
            "[1]MS:1000894|retention time=31.500000\n\
             [1]UO:0000000|unit=UO:0000031|minute\n"
        ));
    }

    #[test]
    fn library_constants_live_in_an_all_set_rather_than_in_every_entry() {
        let text = rendered(true);
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
        let text = rendered(true);
        assert!(text.contains("MS:1000885|protein accession=P1\nMS:1000885|protein accession=P2\n"));
    }
}
