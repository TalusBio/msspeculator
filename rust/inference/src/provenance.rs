//! What produced a library, recorded so it can be regenerated.
//!
//! Serialized into an mzSpecLib header and into the sidecar beside a written library. Its own
//! module because it shares no data and no vocabulary with digestion, batching, or thread
//! orchestration, which is the rest of [`crate::library`].

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use msspeculator_core::Artifact;

use crate::library::{LibraryStats, StreamOptions};

/// What produced a library, and from what.
///
/// Serialized whole into an mzSpecLib header and into the sidecar beside a written library, so
/// the copy inside the file and the copy next to it cannot disagree. Field names are the JSON
/// keys; renaming one changes published output, which `sidecar_keys_are_stable` pins.
///
/// Two ways of saying a value is absent, and they mean different things:
///
/// - **`null`** is a knob nobody set. `context.ms`, `context.chrom`, `retention.raw`,
///   `fragments.max_fragments` and `output` all serialize this way, so a reader can see the
///   question was considered and declined.
/// - **omitted** is not known yet. Only `output.counts`, which exists once the last spectrum has
///   been written.
///
/// A new field picks one of those two deliberately.
#[derive(Clone, Debug, serde::Serialize)]
#[non_exhaustive]
pub struct LibraryProvenance {
    pub generator: Generator,
    pub inputs: Inputs,
    pub digestion: Digestion,
    pub modifications: Modifications,
    pub context: Contexts,
    pub retention: Retention,
    pub fragments: FragmentPolicy,
    pub decoys: DecoyPolicy,
    /// Absent when a caller supplied its own sink: there is then no path, no suffix-chosen
    /// format, and no compression, and inventing values for them would put claims in the
    /// provenance that nothing produced.
    pub output: Option<Output>,
}

/// Which build of which tool wrote this.
#[derive(Clone, Debug, serde::Serialize)]
#[non_exhaustive]
pub struct Generator {
    pub tool: &'static str,
    pub version: &'static str,
    pub commit: &'static str,
}

/// The two inputs, each with the digest of the bytes actually read.
#[derive(Clone, Debug, serde::Serialize)]
#[non_exhaustive]
pub struct Inputs {
    /// The spec as asked for, which for a bundled model is a name rather than a path. A temp path
    /// is not an identity; the digest is, and it is computed from the loaded bytes either way.
    pub model: String,
    pub model_blake2b_256: String,
    pub fasta: String,
    pub fasta_blake2b_256: String,
}

#[derive(Clone, Debug, serde::Serialize)]
#[non_exhaustive]
pub struct Digestion {
    pub enzyme: &'static str,
    pub missed_cleavages: usize,
    pub min_length: usize,
    pub max_length: usize,
    pub min_charge: i64,
    pub max_charge: i64,
}

#[derive(Clone, Debug, serde::Serialize)]
#[non_exhaustive]
pub struct Modifications {
    pub fixed: Vec<String>,
    pub variable: Vec<String>,
    pub max_variable_mods: usize,
}

#[derive(Clone, Debug, serde::Serialize)]
#[non_exhaustive]
pub struct Contexts {
    pub ms: Option<MsContextProvenance>,
    pub chrom: Option<String>,
}

/// The acquisition context as requested: a fitted setup by name, or factors spelled out.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(untagged)]
pub enum MsContextProvenance {
    Named {
        setup: String,
    },
    Factors {
        instrument: String,
        detector: String,
        fragmentation: String,
        energy: Option<f32>,
    },
}

/// What the retention numbers in this library actually are.
#[derive(Clone, Debug, serde::Serialize)]
#[non_exhaustive]
pub struct Retention {
    pub normalized: NormalizedRetention,
    /// Present only under a chromatography context, where `rt` is a duration instead of an index.
    pub raw: Option<RawRetention>,
}

/// The normalized value is an index, not a duration, even though the vocabulary makes us declare
/// it in minutes; so the file says so in plain text rather than leaving a reader to infer it from
/// a unit that cannot be right.
#[derive(Clone, Debug, serde::Serialize)]
#[non_exhaustive]
pub struct NormalizedRetention {
    pub term: &'static str,
    pub kind: &'static str,
    pub scale: &'static str,
    pub anchor_check: AnchorCheck,
}

/// Which index is a property of the corpus the model trained on, so the scale is stated as the
/// convention that defines it and then checked: predicting the two anchors says whether this
/// artifact is on that scale. An artifact from another corpus reports `on_scale: false` and its
/// own numbers rather than inheriting a claim that happens to be true of ours.
#[derive(Clone, Debug, serde::Serialize)]
#[non_exhaustive]
pub struct AnchorCheck {
    pub on_scale: bool,
    pub max_abs_error: f64,
    pub anchors: Vec<Anchor>,
}

#[derive(Clone, Debug, serde::Serialize)]
#[non_exhaustive]
pub struct Anchor {
    pub peptide: String,
    pub expected: f64,
    pub predicted: f64,
}

#[derive(Clone, Debug, serde::Serialize)]
#[non_exhaustive]
pub struct RawRetention {
    pub term: &'static str,
    pub unit: &'static str,
    pub chrom_context: String,
}

#[derive(Clone, Debug, serde::Serialize)]
#[non_exhaustive]
pub struct FragmentPolicy {
    pub min_intensity: f64,
    pub max_fragments: Option<usize>,
}

#[derive(Clone, Debug, serde::Serialize)]
#[non_exhaustive]
pub struct DecoyPolicy {
    pub enabled: bool,
    pub method: &'static str,
    pub protein_prefix: &'static str,
    pub collision_policy: &'static str,
}

/// Where the library went.
#[derive(Clone, Debug, serde::Serialize)]
#[non_exhaustive]
pub struct Output {
    pub path: String,
    pub format: &'static str,
    pub compressed: bool,
    /// Absent until the last spectrum is written, since that is when they are known. Flattened,
    /// so they sit beside `path` in the document rather than under a nested key.
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub counts: Option<Counts>,
    /// Absent until the run finishes, for the same reason as the counts and flattened the same
    /// way.
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub timing: Option<Timing>,
}

/// What the run produced. One struct rather than four `Option` fields set together, so "counts
/// exist only after the last spectrum" is stated once instead of four times.
#[derive(Clone, Debug, serde::Serialize)]
#[non_exhaustive]
pub struct Counts {
    pub proteins: usize,
    pub peptides: usize,
    pub precursors: usize,
    pub fragments: usize,
    pub decoys: usize,
}

/// How long the run took, split where digestion hands off to prediction.
///
/// Recorded beside the counts because a progress bar showing it is gone the moment the terminal
/// scrolls, and "this build took 70 seconds, the last one took 700" is a question asked weeks
/// later. Rounded to milliseconds: nothing downstream can act on a finer number, and full `f64`
/// seconds would churn the sidecar's text on every rebuild.
#[derive(Clone, Debug, serde::Serialize)]
#[non_exhaustive]
pub struct Timing {
    pub seconds_digesting: f64,
    pub seconds_predicting: f64,
}

fn seconds(duration: std::time::Duration) -> f64 {
    (duration.as_secs_f64() * 1000.0).round() / 1000.0
}

impl From<&LibraryStats> for Timing {
    fn from(stats: &LibraryStats) -> Self {
        Self {
            seconds_digesting: seconds(stats.digest),
            seconds_predicting: seconds(stats.predict),
        }
    }
}

impl From<&LibraryStats> for Counts {
    fn from(stats: &LibraryStats) -> Self {
        Self {
            proteins: stats.proteins,
            peptides: stats.peptides,
            precursors: stats.precursors,
            fragments: stats.fragments,
            decoys: stats.decoys,
        }
    }
}

impl LibraryProvenance {
    /// Flatten to JSON, which is how both the mzSpecLib header and the sidecar carry it.
    ///
    /// The `Value` round trip is load-bearing, not an intermediate to optimize away: a
    /// `serde_json::Value` map is a `BTreeMap`, so this emits keys sorted. Serializing the struct
    /// straight to a string would emit declaration order and silently rewrite the key order of
    /// every published sidecar and mzSpecLib header.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("provenance is plain data and always serializes")
    }
}

/// blake2b-256 of a file's bytes, hex, streamed so a multi-GB input is not held in memory.
///
/// Identity rather than integrity: two libraries generated from the same digests and the same
/// settings are the same library, which is what makes a published one reproducible.
fn file_digest(path: &Path) -> Result<String> {
    use blake2::digest::{Update, VariableOutput};
    let mut file = File::open(path).with_context(|| format!("hashing {}", path.display()))?;
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
/// explicitly. Counts are not here; they are only known once the library is written, so
/// [`write_sidecar`] appends them. A library whose provenance lives only in a shell history is one
/// nobody can regenerate or trust, which is why this value is both embedded (mzSpecLib) and
/// written beside the library (sidecar) rather than reassembled per format.
pub(crate) fn resolve_provenance(
    opts: &StreamOptions<'_>,
    output: Option<Output>,
    artifact: &Artifact,
    model_digest: &str,
) -> Result<LibraryProvenance> {
    let anchors = msspeculator_core::landmarks::check_retention_scale(artifact)?;
    Ok(LibraryProvenance {
        generator: Generator {
            tool: "msspeculator-cli library",
            version: env!("CARGO_PKG_VERSION"),
            commit: env!("MSSPECULATOR_GIT_COMMIT"),
        },
        inputs: Inputs {
            model: opts.model.spec(),
            model_blake2b_256: model_digest.to_string(),
            fasta: opts.fasta.display().to_string(),
            fasta_blake2b_256: file_digest(opts.fasta)?,
        },
        digestion: Digestion {
            enzyme: "trypsin",
            missed_cleavages: opts.missed_cleavages,
            min_length: opts.min_length,
            max_length: opts.max_length,
            min_charge: opts.min_charge,
            max_charge: opts.max_charge,
        },
        modifications: Modifications {
            fixed: opts.fixed_mods.to_vec(),
            variable: opts.variable_mods.to_vec(),
            max_variable_mods: opts.max_variable_mods,
        },
        context: Contexts {
            ms: opts.ms_context.map(|context| match context {
                msspeculator_core::MsContext::Named(name) => MsContextProvenance::Named {
                    setup: name.clone(),
                },
                msspeculator_core::MsContext::Factors {
                    instrument,
                    detector,
                    fragmentation,
                    energy,
                } => MsContextProvenance::Factors {
                    instrument: instrument.clone(),
                    detector: detector.clone(),
                    fragmentation: fragmentation.clone(),
                    energy: *energy,
                },
            }),
            chrom: opts.chrom_context.map(str::to_string),
        },
        retention: Retention {
            normalized: NormalizedRetention {
                term: "MS:1000896|normalized retention time",
                kind: "dimensionless index, minutes-like",
                scale: msspeculator_core::landmarks::SCALE_DESCRIPTION,
                anchor_check: AnchorCheck {
                    on_scale: anchors.on_scale(),
                    max_abs_error: anchors.max_abs_error,
                    anchors: anchors
                        .anchors
                        .iter()
                        .map(|(peptide, expected, predicted)| Anchor {
                            peptide: peptide.to_string(),
                            expected: *expected,
                            predicted: *predicted,
                        })
                        .collect(),
                },
            },
            raw: opts.chrom_context.map(|name| RawRetention {
                term: "MS:1000894|retention time",
                unit: "minute",
                chrom_context: name.to_string(),
            }),
        },
        fragments: FragmentPolicy {
            min_intensity: opts.min_intensity,
            max_fragments: opts.max_fragments,
        },
        decoys: DecoyPolicy {
            enabled: opts.generate_decoys,
            method: "pseudo-reverse",
            protein_prefix: "DECOY_",
            collision_policy: "skip_if_stripped_sequence_is_a_target",
        },
        output,
    })
}

/// Write the provenance, plus the counts that came out, beside the library.
pub(crate) fn write_sidecar(
    path: &Path,
    provenance: &LibraryProvenance,
    stats: &LibraryStats,
) -> Result<()> {
    // The counts are only known once the last spectrum is written, so they are filled in here
    // rather than left as a second document a reader has to join against the first.
    let mut provenance = provenance.clone();
    if let Some(output) = provenance.output.as_mut() {
        output.counts = Some(stats.into());
        output.timing = Some(stats.into());
    }
    std::fs::write(
        path,
        serde_json::to_string_pretty(&provenance.to_json())? + "\n",
    )
    .with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> LibraryProvenance {
        LibraryProvenance {
            generator: Generator {
                tool: "msspeculator-cli library",
                version: "0.1.0",
                commit: "abc123",
            },
            inputs: Inputs {
                model: "builtin:small-v0".into(),
                model_blake2b_256: "0".repeat(64),
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
                fixed: vec!["C[UNIMOD:4]".into()],
                variable: vec!["M[UNIMOD:35]".into()],
                max_variable_mods: 1,
            },
            context: Contexts {
                ms: Some(MsContextProvenance::Factors {
                    instrument: "Lumos".into(),
                    detector: "FTMS".into(),
                    fragmentation: "HCD".into(),
                    energy: Some(27.0),
                }),
                chrom: Some("dsA".into()),
            },
            retention: Retention {
                normalized: NormalizedRetention {
                    term: "MS:1000896|normalized retention time",
                    kind: "dimensionless index, minutes-like",
                    scale: "anchored",
                    anchor_check: AnchorCheck {
                        on_scale: true,
                        max_abs_error: 0.5,
                        anchors: vec![Anchor {
                            peptide: "TFAHTESHISK".into(),
                            expected: 0.0,
                            predicted: 0.1,
                        }],
                    },
                },
                raw: Some(RawRetention {
                    term: "MS:1000894|retention time",
                    unit: "minute",
                    chrom_context: "dsA".into(),
                }),
            },
            fragments: FragmentPolicy {
                min_intensity: 0.01,
                max_fragments: Some(15),
            },
            decoys: DecoyPolicy {
                enabled: true,
                method: "pseudo-reverse",
                protein_prefix: "DECOY_",
                collision_policy: "skip",
            },
            output: Some(Output {
                path: "library.tsv".into(),
                format: "diann-tsv",
                compressed: false,
                counts: None,
                timing: None,
            }),
        }
    }

    /// Every dotted key of a fully populated provenance, in the order it is published.
    fn keys(value: &serde_json::Value, prefix: &str, out: &mut Vec<String>) {
        match value {
            serde_json::Value::Object(fields) => {
                for (key, value) in fields {
                    let path = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{prefix}.{key}")
                    };
                    keys(value, &path, out);
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    keys(item, &format!("{prefix}[]"), out);
                }
            }
            _ => out.push(prefix.to_string()),
        }
    }

    /// These names are published: they land in every sidecar and in every mzSpecLib header, where
    /// a reader matches on them. Renaming a struct field silently rewrites them, so the list is
    /// written out once here rather than left implicit in thirteen struct definitions.
    ///
    /// Sorted, because `to_json` goes through a `serde_json::Value`, whose map is a `BTreeMap`.
    /// A change to declaration order must not move a key.
    #[test]
    fn sidecar_keys_are_stable() {
        let mut found = Vec::new();
        keys(&sample().to_json(), "", &mut found);
        assert_eq!(
            found,
            vec![
                "context.chrom",
                "context.ms.detector",
                "context.ms.energy",
                "context.ms.fragmentation",
                "context.ms.instrument",
                "decoys.collision_policy",
                "decoys.enabled",
                "decoys.method",
                "decoys.protein_prefix",
                "digestion.enzyme",
                "digestion.max_charge",
                "digestion.max_length",
                "digestion.min_charge",
                "digestion.min_length",
                "digestion.missed_cleavages",
                "fragments.max_fragments",
                "fragments.min_intensity",
                "generator.commit",
                "generator.tool",
                "generator.version",
                "inputs.fasta",
                "inputs.fasta_blake2b_256",
                "inputs.model",
                "inputs.model_blake2b_256",
                "modifications.fixed[]",
                "modifications.max_variable_mods",
                "modifications.variable[]",
                "output.compressed",
                "output.format",
                "output.path",
                "retention.normalized.anchor_check.anchors[].expected",
                "retention.normalized.anchor_check.anchors[].peptide",
                "retention.normalized.anchor_check.anchors[].predicted",
                "retention.normalized.anchor_check.max_abs_error",
                "retention.normalized.anchor_check.on_scale",
                "retention.normalized.kind",
                "retention.normalized.scale",
                "retention.normalized.term",
                "retention.raw.chrom_context",
                "retention.raw.term",
                "retention.raw.unit",
            ]
        );
    }

    /// A knob nobody set is null; something not known yet is omitted. Both are absent to a reader
    /// scanning for a value, and they are not the same claim.
    #[test]
    fn unset_knobs_are_null_and_unknown_counts_are_omitted() {
        let mut bare = sample();
        bare.context = Contexts {
            ms: None,
            chrom: None,
        };
        bare.retention.raw = None;
        bare.fragments.max_fragments = None;
        let json = bare.to_json();
        for key in ["/context/ms", "/context/chrom", "/retention/raw"] {
            assert_eq!(json.pointer(key), Some(&serde_json::Value::Null), "{key}");
        }
        assert_eq!(
            json.pointer("/fragments/max_fragments"),
            Some(&serde_json::Value::Null)
        );
        // Counts are absent entirely, not null: they are not a question anyone declined.
        assert!(json.pointer("/output/precursors").is_none());

        // And a caller-supplied sink has no output block at all.
        let mut streamed = sample();
        streamed.output = None;
        assert_eq!(
            streamed.to_json().pointer("/output"),
            Some(&serde_json::Value::Null)
        );
    }

    #[test]
    fn the_sidecar_reports_every_count_including_decoys() {
        let stats = LibraryStats {
            proteins: 1,
            peptides: 2,
            precursors: 3,
            fragments: 4,
            decoys: 5,
            ..LibraryStats::default()
        };
        let counts = Counts::from(&stats);
        let json = serde_json::to_value(&counts).unwrap();
        for (key, expected) in [
            ("proteins", 1),
            ("peptides", 2),
            ("precursors", 3),
            ("fragments", 4),
            ("decoys", 5),
        ] {
            assert_eq!(json[key], expected, "{key}");
        }
    }

    /// The durable half of what a progress bar showed: the bar is gone when the terminal
    /// scrolls, and "this build took 70 seconds, the last one took 700" gets asked weeks later.
    #[test]
    fn the_sidecar_records_each_phase_to_the_millisecond() {
        let stats = LibraryStats {
            digest: std::time::Duration::from_micros(1_500_400),
            predict: std::time::Duration::from_micros(70_000_999),
            ..LibraryStats::default()
        };
        let json = serde_json::to_value(Timing::from(&stats)).unwrap();
        assert_eq!(json["seconds_digesting"], 1.5);
        assert_eq!(json["seconds_predicting"], 70.001);
    }
}
