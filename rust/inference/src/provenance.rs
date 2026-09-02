//! What produced a library, recorded so it can be regenerated.
//!
//! Serialized into an mzSpecLib header and into the sidecar beside a written library. Its own
//! module because it shares no data and no vocabulary with digestion, batching, or thread
//! orchestration, which is the rest of [`crate::library`].

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use msspeculator_core::{Artifact, ModelSource};

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
/// - **omitted** is not known yet, or was never measured here. `output.counts` exists once the
///   last spectrum has been written; `retention.normalized.anchor_check` exists only where an
///   artifact was loaded to predict the anchors, which a build always does and a reader
///   verifying one never does.
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
    /// Absent when nothing predicted the anchors, which is the case for every provenance
    /// resolved from settings alone. A build always fills it, so a written library always
    /// carries it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anchor_check: Option<AnchorCheck>,
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

/// How long the run took, one number per phase the build reported.
///
/// Recorded beside the counts because a progress bar showing it is gone the moment the terminal
/// scrolls, and "this build took 70 seconds, the last one took 700" is a question asked weeks
/// later. Rounded to milliseconds: nothing downstream can act on a finer number, and full `f64`
/// seconds would churn the sidecar's text on every rebuild.
///
/// One field per phase rather than two, because the three scale on unrelated things: digestion on
/// the proteome, prediction on the precursor count and the model, loading on the artifact and on
/// whether the page cache is cold. The three are disjoint and sum to the build.
#[derive(Clone, Debug, serde::Serialize)]
#[non_exhaustive]
pub struct Timing {
    pub seconds_digesting: f64,
    pub seconds_loading: f64,
    pub seconds_predicting: f64,
}

fn seconds(duration: std::time::Duration) -> f64 {
    (duration.as_secs_f64() * 1000.0).round() / 1000.0
}

impl From<&LibraryStats> for Timing {
    fn from(stats: &LibraryStats) -> Self {
        Self {
            seconds_digesting: seconds(stats.digest),
            seconds_loading: seconds(stats.load),
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
    resolve(
        opts,
        output,
        model_digest,
        Some(AnchorCheck {
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
        }),
    )
}

/// Everything a configuration determines, with the two things it does not passed in.
///
/// The anchor check is a measurement of a loaded artifact rather than a resolved setting, and the
/// model digest comes from bytes somebody else read; splitting them out is what lets
/// [`check_library`] answer the same question without loading a model.
fn resolve(
    opts: &StreamOptions<'_>,
    output: Option<Output>,
    model_digest: &str,
    anchor_check: Option<AnchorCheck>,
) -> Result<LibraryProvenance> {
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
                anchor_check,
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

/// Where the sidecar for a library goes by default: `lib.tsv` -> `lib.tsv.config.json`.
///
/// Appended to the whole name rather than replacing the extension, which
/// `Path::with_extension` would do: `lib.mzspeclib.config.json` has lost which format the
/// sidecar describes. Built as an OS string, since a path need not be UTF-8.
///
/// Public because it is the convention that lets [`check_library`] find the provenance of a
/// library whose format has no header to carry it.
pub fn sidecar_path(library: &Path) -> std::path::PathBuf {
    let mut name = library.to_path_buf().into_os_string();
    name.push(".config.json");
    std::path::PathBuf::from(name)
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

/// Whether a library on disk was built the way a set of options describes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LibraryCheck {
    /// Same keys, same values: a build with these options would produce this library.
    Same,
    /// Built differently. Never empty: a key only one side records is a difference like any
    /// other, since `--max-fragments` dropped from a rebuild is exactly as much a change as one
    /// given a new value.
    Different { differences: Vec<Difference> },
    /// Nothing to compare against: no `msspeculator:` attributes in the header and no sidecar
    /// beside it. Another tool's library, or one of ours written with `--no-config-out` in a
    /// format that carries no header. Not a mismatch.
    Unknown,
}

/// One provenance key that a library and a set of options spell differently.
///
/// Either side can be `None`, and it means the same thing on both: nothing was recorded under
/// that key. An unset knob is dropped from a header rather than written as a null, so a setting
/// present in one build and absent from the other shows up here as a missing side.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Difference {
    /// The provenance key, as the header spells it without the `msspeculator:` prefix.
    pub key: String,
    /// What the library on disk says.
    pub library: Option<String>,
    /// What a build with these options would have recorded.
    pub expected: Option<String>,
}

/// Whether the library at `path` was built the way `opts` describes.
///
/// For a search reading someone else's library: warn per [`Difference`], naming both values, and
/// run anyway. For a build about to overwrite one: [`LibraryCheck::Same`] means the file already
/// is what this run would produce.
///
/// Reads the header and stops at the first spectrum, so the library itself costs one open
/// whatever its size. A format with no header to carry provenance falls back to the
/// [`sidecar_path`] beside it. The other side is resolved from `opts` without loading a model,
/// but it does hash the FASTA, and that is a full read of it.
///
/// Only what the settings determine is compared. Which build wrote the library, the paths its
/// inputs were named by, where it was written and how its retention anchors measured are all
/// left out: none of them is a disagreement about how the library was built.
pub fn check_library(path: &Path, opts: &StreamOptions<'_>) -> Result<LibraryCheck> {
    // The header first, because that copy cannot be separated from the library it describes. A
    // DIA-NN TSV has no header to put it in, so for that format the sidecar is the only copy.
    let mut recorded = crate::mzspeclib::read_header_attributes(path)?;
    if recorded.is_empty() {
        recorded = sidecar_attributes(&sidecar_path(path));
    }
    if recorded.is_empty() {
        return Ok(LibraryCheck::Unknown);
    }
    let library = comparable(recorded);
    let expected = comparable(crate::mzspeclib::attributes(&expected_provenance(opts)?));
    // Over the union, so that `Same` is a claim about every key either side records rather than
    // only the ones they happen to share. A key one side has alone is a difference: it is either
    // a knob set in one build and not the other, or a library from a version that recorded
    // something this one does not, and both are answers a reader wants rather than silence.
    let differences = library
        .keys()
        .chain(expected.keys())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .filter(|key| library.get(*key) != expected.get(*key))
        .map(|key| Difference {
            key: key.clone(),
            library: library.get(key).cloned(),
            expected: expected.get(key).cloned(),
        })
        .collect::<Vec<_>>();
    if differences.is_empty() {
        Ok(LibraryCheck::Same)
    } else {
        Ok(LibraryCheck::Different { differences })
    }
}

/// The provenance beside a library, in the spelling a header would have carried it.
///
/// Nothing here is an error: a missing sidecar is the ordinary case, and a file at that name
/// that is not one of ours says nothing about the library either way. Both come back as no
/// provenance, because a check that refuses to run is worse than a check that reports it has
/// nothing to compare.
fn sidecar_attributes(path: &Path) -> BTreeMap<String, String> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .map(|config| crate::mzspeclib::attributes_from_json(&config))
        .unwrap_or_default()
}

/// The provenance a build with these settings would record, minus what settings cannot say.
///
/// No model load: a bundled artifact's digest is a constant and a file's is its bytes, so
/// neither needs the tensors. No anchor check and no output, both of which say nothing when
/// nothing was predicted and nothing was written.
fn expected_provenance(opts: &StreamOptions<'_>) -> Result<LibraryProvenance> {
    let digest = match &opts.model {
        ModelSource::Builtin(model) => msspeculator_core::builtin_digest(*model)?.to_string(),
        ModelSource::File(path) => file_digest(path)?,
    };
    resolve(opts, None, &digest, None)
}

/// The keys worth comparing, of the ones a header carries.
///
/// Dropped, each because a difference in it is not a disagreement about how the library was
/// built:
///
/// - `generator.*` is which build wrote the file. The same settings through a later release are
///   still the same library, and comparing this would warn on every upgrade.
/// - `inputs.model` and `inputs.fasta` are paths as they were typed. `../human.fasta` and
///   `/data/human.fasta` are one file; the digest beside each of them is the identity, and it is
///   kept.
/// - `output.*` is where the library went, which nobody reading one is claiming.
/// - `retention.normalized.anchor_check.*` is measured from a loaded artifact rather than
///   resolved from settings, so only a build has it — and a different model is already visible
///   as `inputs.model_blake2b_256`.
fn comparable(pairs: BTreeMap<String, String>) -> BTreeMap<String, String> {
    pairs
        .into_iter()
        .filter(|(key, _)| {
            !(key.starts_with("generator.")
                || key.starts_with("output.")
                || key.starts_with("retention.normalized.anchor_check.")
                || key == "inputs.model"
                || key == "inputs.fasta")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scratch::Scratch;

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
                    anchor_check: Some(AnchorCheck {
                        on_scale: true,
                        max_abs_error: 0.5,
                        anchors: vec![Anchor {
                            peptide: "TFAHTESHISK".into(),
                            expected: 0.0,
                            predicted: 0.1,
                        }],
                    }),
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

    fn options(fasta: &Path) -> StreamOptions<'_> {
        StreamOptions {
            model: ModelSource::Builtin(msspeculator_core::BuiltinModel::SmallV0),
            fasta,
            activation: None,
            ms_context: None,
            chrom_context: None,
            min_intensity: 0.01,
            missed_cleavages: 2,
            min_length: 7,
            max_length: 30,
            min_charge: 2,
            max_charge: 4,
            fixed_mods: &[],
            variable_mods: &[],
            max_variable_mods: 1,
            max_fragments: None,
            generate_decoys: false,
            progress: None,
        }
    }

    /// The provenance a real build resolves: an output block, and an anchor check measured from
    /// a loaded artifact. Both are things a reader checking the library cannot have.
    fn built(path: &Path, opts: &StreamOptions<'_>) -> LibraryProvenance {
        resolve(
            opts,
            Some(Output {
                path: path.display().to_string(),
                format: "mzspeclib-text",
                compressed: false,
                counts: None,
                timing: None,
            }),
            msspeculator_core::builtin_digest(msspeculator_core::BuiltinModel::SmallV0).unwrap(),
            Some(AnchorCheck {
                on_scale: true,
                max_abs_error: 0.5,
                anchors: Vec::new(),
            }),
        )
        .unwrap()
    }

    fn write_header(path: &Path, opts: &StreamOptions<'_>) {
        let provenance = built(path, opts);
        let file = File::create(path).unwrap();
        let mut sink = crate::mzspeclib::MzSpecLibSink::new(file, path);
        crate::library::LibrarySink::header(&mut sink, &provenance).unwrap();
    }

    #[test]
    fn a_library_these_options_would_produce_is_the_same_library() {
        let fasta = Scratch::holding("check.fasta", ">P1\nPEPTIDEK\n");
        let library = Scratch::new("check.mzspeclib.txt");
        write_header(library.path(), &options(fasta.path()));
        assert_eq!(
            check_library(library.path(), &options(fasta.path())).unwrap(),
            LibraryCheck::Same,
            "a path, a format and an anchor measurement are not settings"
        );
    }

    #[test]
    fn a_changed_setting_is_the_only_difference() {
        let fasta = Scratch::holding("changed.fasta", ">P1\nPEPTIDEK\n");
        let library = Scratch::new("changed.mzspeclib.txt");
        write_header(library.path(), &options(fasta.path()));
        let searching = StreamOptions {
            missed_cleavages: 3,
            ..options(fasta.path())
        };
        assert_eq!(
            check_library(library.path(), &searching).unwrap(),
            LibraryCheck::Different {
                differences: vec![Difference {
                    key: "digestion.missed_cleavages".into(),
                    library: Some("2".into()),
                    expected: Some("3".into()),
                }]
            }
        );
    }

    /// The headline case: same settings, another proteome. The digest carries it, and the path it
    /// was typed as does not enter into it.
    #[test]
    fn another_fasta_differs_by_digest_and_not_by_path() {
        let built = Scratch::holding("built.fasta", ">P1\nPEPTIDEK\n");
        let searched = Scratch::holding("searched.fasta", ">P1\nPEPTIDEKK\n");
        let library = Scratch::new("fasta.mzspeclib.txt");
        write_header(library.path(), &options(built.path()));
        let LibraryCheck::Different { differences } =
            check_library(library.path(), &options(searched.path())).unwrap()
        else {
            panic!("a library from another proteome is not the same library");
        };
        let keys: Vec<&str> = differences.iter().map(|d| d.key.as_str()).collect();
        assert_eq!(keys, vec!["inputs.fasta_blake2b_256"]);
    }

    /// A knob set in one build and left alone in the other is dropped from one header and not
    /// the other, so comparing only the keys both sides carry would call these the same library.
    #[test]
    fn a_setting_dropped_from_a_rebuild_is_a_difference() {
        let fasta = Scratch::holding("dropped.fasta", ">P1\nPEPTIDEK\n");
        let library = Scratch::new("dropped.mzspeclib.txt");
        write_header(
            library.path(),
            &StreamOptions {
                max_fragments: Some(15),
                ..options(fasta.path())
            },
        );
        assert_eq!(
            check_library(library.path(), &options(fasta.path())).unwrap(),
            LibraryCheck::Different {
                differences: vec![Difference {
                    key: "fragments.max_fragments".into(),
                    library: Some("15".into()),
                    expected: None,
                }]
            }
        );
    }

    #[test]
    fn a_library_without_provenance_is_unknown_rather_than_a_mismatch() {
        let fasta = Scratch::holding("unknown.fasta", ">P1\nPEPTIDEK\n");
        let library = Scratch::holding("unknown.tsv", "PrecursorMz\tProductMz\n100.0\t200.0\n");
        assert_eq!(
            check_library(library.path(), &options(fasta.path())).unwrap(),
            LibraryCheck::Unknown
        );
    }

    /// `.config.json` appends to the whole name, so which format the sidecar describes stays
    /// visible and the two files sort next to each other.
    #[test]
    fn the_sidecar_sits_beside_the_library_it_describes() {
        for (library, expected) in [
            ("lib.tsv", "lib.tsv.config.json"),
            ("lib.mzspeclib.txt", "lib.mzspeclib.txt.config.json"),
            ("lib.mzspeclib.txt.gz", "lib.mzspeclib.txt.gz.config.json"),
            ("no-extension", "no-extension.config.json"),
        ] {
            assert_eq!(
                sidecar_path(Path::new(library)),
                std::path::PathBuf::from(expected),
                "{library}"
            );
        }
    }

    /// A DIA-NN TSV has no header to carry provenance, so the sidecar is the only copy it has —
    /// and the two copies have to compare the same way, nulls and software version included.
    #[test]
    fn a_format_with_no_header_is_checked_against_its_sidecar() {
        let fasta = Scratch::holding("sidecar.fasta", ">P1\nPEPTIDEK\n");
        let library = Scratch::holding("sidecar.tsv", "PrecursorMz\tProductMz\n100.0\t200.0\n");
        // Named after the library rather than handed out by `Scratch`, since the name is the
        // convention under test.
        let beside_it = sidecar_path(library.path());
        write_sidecar(
            &beside_it,
            &built(library.path(), &options(fasta.path())),
            &LibraryStats::default(),
        )
        .unwrap();
        let checked = check_library(library.path(), &options(fasta.path()));
        let _ = std::fs::remove_file(&beside_it);
        assert_eq!(checked.unwrap(), LibraryCheck::Same);
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
            load: std::time::Duration::from_micros(12_300_500),
            predict: std::time::Duration::from_micros(70_000_999),
            ..LibraryStats::default()
        };
        let json = serde_json::to_value(Timing::from(&stats)).unwrap();
        assert_eq!(json["seconds_digesting"], 1.5);
        // The phase that exists to explain a pause is the one whose duration answers "did the
        // rebuild get slower loading the model, or running it?".
        assert_eq!(json["seconds_loading"], 12.301);
        assert_eq!(json["seconds_predicting"], 70.001);
    }
}
