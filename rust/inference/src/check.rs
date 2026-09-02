//! Whether a library on disk was built the way a set of options describes.
//!
//! Reads the provenance a library carries and compares it against the [`Settings`] those options
//! would resolve to. Its own module because it depends on both halves of that sentence —
//! [`crate::provenance`] for the document and [`crate::mzspeclib`] for the header that carries
//! one — and neither of them depends on it.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use anyhow::Result;

use crate::library::StreamOptions;
use crate::provenance::{sidecar_path, Settings};

/// The two settings keys a check ignores.
///
/// [`Settings`] is the whole of what the options decide, so `generator.*`, `output.*` and
/// `retention.*` are left out of the comparison simply by not being in it. These two are the
/// exception: they sit in `inputs` beside the digests, but they are paths as they were typed.
/// `../human.fasta` and `/data/human.fasta` are one file, and the digest beside each of them is
/// the identity.
const NOT_AN_IDENTITY: [&str; 2] = ["inputs.model", "inputs.fasta"];

/// Whether a library on disk was built the way a set of options describes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LibraryCheck {
    /// Same keys, same values: a build with these options would produce this library.
    Same,
    /// Built differently. Never empty: a key only one side records is a difference like any
    /// other, since `--max-fragments` dropped from a rebuild is exactly as much a change as one
    /// given a new value.
    Different(Vec<Difference>),
    /// Nothing to compare against: no `msspeculator:` attributes in the header and no sidecar
    /// where one would be. Another tool's library, one of ours written with `--no-config-out` in
    /// a format that carries no header, or a file this could not read at all. Not a mismatch.
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

impl fmt::Display for Difference {
    /// A key one side never recorded reads as `unset`, which is what an omitted knob is; any
    /// other spelling would be inventing a value for it. Written here rather than at each caller
    /// so the wording lives with the `Option` that permits it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let side = |value: &Option<String>| value.clone().unwrap_or_else(|| "unset".to_string());
        write!(
            f,
            "{} was {}, now {}",
            self.key,
            side(&self.library),
            side(&self.expected)
        )
    }
}

/// Whether the library at `library` was built the way `opts` describes.
///
/// For a search reading someone else's library: warn per [`Difference`], which prints both
/// values, and run anyway. For a build about to overwrite one: [`LibraryCheck::Same`] means the
/// file already is what this run would produce.
///
/// Reads the header and stops at the end of it, so the library itself costs one open whatever its
/// size. A format with no header to carry provenance is compared against the sidecar beside it
/// instead — `sidecar` names it, or `None` for the default [`sidecar_path`]. The other side is
/// resolved without loading a model, but it does hash the FASTA, and that is a full read of it.
///
/// Only what the settings determine is compared: see [`Settings`] for what that is and is not.
/// A library this cannot read the provenance of is [`LibraryCheck::Unknown`] rather than an
/// error — the one thing that fails here is a FASTA or a model that cannot be hashed, because
/// then there is no expected side to compare at all.
pub fn check_library(
    library: &Path,
    sidecar: Option<&Path>,
    opts: &StreamOptions<'_>,
) -> Result<LibraryCheck> {
    opts.validate()?;
    // The header first, because that copy cannot be separated from the library it describes. A
    // DIA-NN TSV has no header to put it in, so for that format the sidecar is the only copy.
    let beside_it = sidecar.map_or_else(|| sidecar_path(library), Path::to_path_buf);
    let Some(recorded) = from_header(library).or_else(|| from_sidecar(&beside_it)) else {
        return Ok(LibraryCheck::Unknown);
    };
    let expected = Settings::resolve(opts, &opts.model.digest()?)?.attributes();
    Ok(compare(comparable(recorded), comparable(expected)))
}

/// The pairs the library's own header carries, or `None` when it carries none of ours.
fn from_header(library: &Path) -> Option<BTreeMap<String, String>> {
    let pairs = crate::library::open_library(library)
        .map(crate::mzspeclib::header_attributes)
        .unwrap_or_default();
    Some(pairs).filter(|pairs| !pairs.is_empty())
}

/// The provenance beside a library, in the spelling a header would have carried it.
///
/// Nothing here is an error: a missing sidecar is the ordinary case, and a file at that name that
/// is not one of ours says nothing about the library either way. Both come back as no provenance,
/// because a check that refuses to run is worse than one that reports it has nothing to compare.
///
/// In the header's spelling deliberately: a library's two copies of its provenance are the same
/// provenance, so whichever one a reader finds has to compare the same way. The sidecar keeps the
/// nulls a header drops, and those fall out here.
fn from_sidecar(path: &Path) -> Option<BTreeMap<String, String>> {
    let text = std::fs::read_to_string(path).ok()?;
    let config = serde_json::from_str(&text).ok()?;
    Some(crate::provenance::flatten(&config)).filter(|pairs| !pairs.is_empty())
}

/// A provenance narrowed to the keys two builds can disagree about.
///
/// An allowlist rather than a list of what to skip, and the allowlist is [`Settings`] itself: a
/// key that no option decides cannot become a difference by being added to the document, and a
/// new setting is compared the moment it exists.
fn comparable(pairs: BTreeMap<String, String>) -> BTreeMap<String, String> {
    pairs
        .into_iter()
        .filter(|(key, _)| Settings::owns(key) && !NOT_AN_IDENTITY.contains(&key.as_str()))
        .collect()
}

/// Over the union, so that `Same` is a claim about every key either side records rather than only
/// the ones they happen to share. A key one side has alone is a difference: it is either a knob
/// set in one build and not the other, or a library from a version that recorded something this
/// one does not, and both are answers a reader wants rather than silence.
fn compare(library: BTreeMap<String, String>, expected: BTreeMap<String, String>) -> LibraryCheck {
    let differences: Vec<Difference> = library
        .keys()
        .chain(expected.keys())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|key| library.get(*key) != expected.get(*key))
        .map(|key| Difference {
            key: key.clone(),
            library: library.get(key).cloned(),
            expected: expected.get(key).cloned(),
        })
        .collect();
    if differences.is_empty() {
        LibraryCheck::Same
    } else {
        LibraryCheck::Different(differences)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::library::{LibrarySink, LibraryStats};
    use crate::provenance::write_sidecar;
    use crate::scratch::Scratch;
    use msspeculator_core::{BuiltinModel, ModelSource};

    fn options(fasta: &Path) -> StreamOptions<'_> {
        StreamOptions {
            model: ModelSource::Builtin(BuiltinModel::SmallV0),
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

    /// A library whose provenance lives only in the sidecar, which is the DIA-NN TSV case.
    ///
    /// The provenance a real build writes: an output block and an anchor check measured from a
    /// loaded artifact, neither of which a reader checking the library can have. Both are
    /// therefore what the comparison has to look past, so both are here rather than defaulted.
    fn built(library: &Scratch, opts: &StreamOptions<'_>) -> Scratch {
        let mut provenance = crate::provenance::tests::sample();
        provenance.settings = Settings::resolve(opts, BuiltinModel::SmallV0.digest()).unwrap();
        let sidecar = Scratch::at(sidecar_path(library.path()));
        write_sidecar(sidecar.path(), &provenance, &LibraryStats::default()).unwrap();
        sidecar
    }

    fn tsv(name: &str) -> Scratch {
        Scratch::holding(name, "PrecursorMz\tProductMz\n100.0\t200.0\n")
    }

    /// What a check would compare these options against, resolved the way `check_library` does —
    /// the model's digest from the model, not from a constant a test chose.
    fn settings(opts: &StreamOptions<'_>) -> BTreeMap<String, String> {
        comparable(
            Settings::resolve(opts, &opts.model.digest().unwrap())
                .unwrap()
                .attributes(),
        )
    }

    /// The promise is "would the same library be generated", not "were the same arguments typed".
    /// So every option that changes what comes out has to change these attributes: a knob missing
    /// from [`Settings`] is a knob two libraries can differ by while comparing equal, which is the
    /// one way this can give a wrong answer rather than no answer.
    ///
    /// `activation` was exactly that knob — it rewrites the artifact's activation and so changes
    /// every peak — until it was recorded. Adding a `StreamOptions` field means adding it here.
    #[test]
    fn every_option_that_changes_the_library_changes_the_settings() {
        let fasta = Scratch::holding("knobs.fasta", ">P1\nPEPTIDEKPEPTIDER\n");
        let other = Scratch::holding("knobs-other.fasta", ">P1\nPEPTIDEKPEPTIDERK\n");
        let weights = Scratch::holding("knobs.safetensors", "not a real artifact, only hashed");
        let base = options(fasta.path());
        let baseline = settings(&base);
        let named = msspeculator_core::MsContext::Named("setup".into());
        let mods = ["C[UNIMOD:4]".to_string()];
        for (knob, changed) in [
            (
                "model",
                StreamOptions {
                    model: ModelSource::File(weights.path().into()),
                    ..options(fasta.path())
                },
            ),
            ("fasta", options(other.path())),
            (
                "activation",
                StreamOptions {
                    activation: Some("relu"),
                    ..options(fasta.path())
                },
            ),
            (
                "ms_context",
                StreamOptions {
                    ms_context: Some(&named),
                    ..options(fasta.path())
                },
            ),
            (
                "chrom_context",
                StreamOptions {
                    chrom_context: Some("dsA"),
                    ..options(fasta.path())
                },
            ),
            (
                "min_intensity",
                StreamOptions {
                    min_intensity: 0.5,
                    ..options(fasta.path())
                },
            ),
            (
                "missed_cleavages",
                StreamOptions {
                    missed_cleavages: 3,
                    ..options(fasta.path())
                },
            ),
            (
                "min_length",
                StreamOptions {
                    min_length: 8,
                    ..options(fasta.path())
                },
            ),
            (
                "max_length",
                StreamOptions {
                    max_length: 25,
                    ..options(fasta.path())
                },
            ),
            (
                "min_charge",
                StreamOptions {
                    min_charge: 3,
                    ..options(fasta.path())
                },
            ),
            (
                "max_charge",
                StreamOptions {
                    max_charge: 5,
                    ..options(fasta.path())
                },
            ),
            (
                "fixed_mods",
                StreamOptions {
                    fixed_mods: &mods,
                    ..options(fasta.path())
                },
            ),
            (
                "variable_mods",
                StreamOptions {
                    variable_mods: &mods,
                    ..options(fasta.path())
                },
            ),
            (
                "max_variable_mods",
                StreamOptions {
                    max_variable_mods: 2,
                    ..options(fasta.path())
                },
            ),
            (
                "max_fragments",
                StreamOptions {
                    max_fragments: Some(8),
                    ..options(fasta.path())
                },
            ),
            (
                "generate_decoys",
                StreamOptions {
                    generate_decoys: true,
                    ..options(fasta.path())
                },
            ),
        ] {
            assert_ne!(
                settings(&changed),
                baseline,
                "{knob} changes the library but not what a check compares"
            );
        }
        // `progress` is the one option that does not, because it changes nothing on the way out.
        let watched = StreamOptions {
            progress: Some(&|_| {}),
            ..options(fasta.path())
        };
        assert_eq!(settings(&watched), baseline, "progress is not a setting");
    }

    /// Four of the compared keys are constants rather than options. They are compared because a
    /// library built when `collision_policy` meant something else *is* a different library — which
    /// makes them published values, not documentation. Rewording one flags every library ever
    /// built, so it has to be a deliberate act with this test to change.
    #[test]
    fn the_compared_constants_are_published_values() {
        let fasta = Scratch::holding("constants.fasta", ">P1\nPEPTIDEK\n");
        let settings = settings(&options(fasta.path()));
        for (key, value) in [
            ("digestion.enzyme", "trypsin"),
            ("decoys.method", "pseudo-reverse"),
            ("decoys.protein_prefix", "DECOY_"),
            (
                "decoys.collision_policy",
                "skip_if_stripped_sequence_is_a_target",
            ),
        ] {
            assert_eq!(settings.get(key).map(String::as_str), Some(value), "{key}");
        }
    }

    #[test]
    fn a_library_these_options_would_produce_is_the_same_library() {
        let fasta = Scratch::holding("check.fasta", ">P1\nPEPTIDEK\n");
        let library = tsv("check.tsv");
        let _sidecar = built(&library, &options(fasta.path()));
        assert_eq!(
            check_library(library.path(), None, &options(fasta.path())).unwrap(),
            LibraryCheck::Same,
            "a release, a path, a format and an anchor measurement are not settings"
        );
    }

    #[test]
    fn a_changed_setting_is_the_only_difference() {
        let fasta = Scratch::holding("changed.fasta", ">P1\nPEPTIDEK\n");
        let library = tsv("changed.tsv");
        let _sidecar = built(&library, &options(fasta.path()));
        let searching = StreamOptions {
            missed_cleavages: 3,
            ..options(fasta.path())
        };
        assert_eq!(
            check_library(library.path(), None, &searching).unwrap(),
            LibraryCheck::Different(vec![Difference {
                key: "digestion.missed_cleavages".into(),
                library: Some("2".into()),
                expected: Some("3".into()),
            }])
        );
    }

    /// The headline case: same settings, another proteome. The digest carries it, and the path it
    /// was typed as does not enter into it.
    #[test]
    fn another_fasta_differs_by_digest_and_not_by_path() {
        let original = Scratch::holding("built.fasta", ">P1\nPEPTIDEK\n");
        let searched = Scratch::holding("searched.fasta", ">P1\nPEPTIDEKK\n");
        let library = tsv("fasta.tsv");
        let _sidecar = built(&library, &options(original.path()));
        let LibraryCheck::Different(differences) =
            check_library(library.path(), None, &options(searched.path())).unwrap()
        else {
            panic!("a library from another proteome is not the same library");
        };
        let keys: Vec<&str> = differences.iter().map(|d| d.key.as_str()).collect();
        assert_eq!(keys, vec!["inputs.fasta_blake2b_256"]);
    }

    /// A knob set in one build and left alone in the other is dropped from one header and not the
    /// other, so comparing only the keys both sides carry would call these the same library.
    #[test]
    fn a_setting_dropped_from_a_rebuild_is_a_difference() {
        let fasta = Scratch::holding("dropped.fasta", ">P1\nPEPTIDEK\n");
        let library = tsv("dropped.tsv");
        let _sidecar = built(
            &library,
            &StreamOptions {
                max_fragments: Some(15),
                ..options(fasta.path())
            },
        );
        let difference = Difference {
            key: "fragments.max_fragments".into(),
            library: Some("15".into()),
            expected: None,
        };
        assert_eq!(
            check_library(library.path(), None, &options(fasta.path())).unwrap(),
            LibraryCheck::Different(vec![difference.clone()])
        );
        assert_eq!(
            difference.to_string(),
            "fragments.max_fragments was 15, now unset"
        );
    }

    /// The header is the copy that cannot be separated from the library, so it wins when both
    /// exist — and it has to compare the same way the sidecar does, nulls and software version
    /// included.
    #[test]
    fn a_header_is_read_in_place_of_a_sidecar_and_agrees_with_one() {
        let fasta = Scratch::holding("header.fasta", ">P1\nPEPTIDEK\n");
        let library = Scratch::new("header.mzspeclib.txt");
        let mut provenance = crate::provenance::tests::sample();
        provenance.settings =
            Settings::resolve(&options(fasta.path()), BuiltinModel::SmallV0.digest()).unwrap();
        let mut sink = crate::mzspeclib::MzSpecLibSink::new(
            std::fs::File::create(library.path()).unwrap(),
            library.path(),
        );
        sink.header(&provenance).unwrap();
        drop(sink);
        assert_eq!(
            check_library(library.path(), None, &options(fasta.path())).unwrap(),
            LibraryCheck::Same
        );
    }

    #[test]
    fn a_library_without_provenance_is_unknown_rather_than_a_mismatch() {
        let fasta = Scratch::holding("unknown.fasta", ">P1\nPEPTIDEK\n");
        let library = tsv("unknown.tsv");
        assert_eq!(
            check_library(library.path(), None, &options(fasta.path())).unwrap(),
            LibraryCheck::Unknown
        );
    }

    /// Every way a library can fail to answer the question is the same answer: nothing to compare.
    /// A check that raises instead is a check that can fail a build it was only advising.
    #[test]
    fn nothing_readable_is_unknown_rather_than_an_error() {
        let fasta = Scratch::holding("unreadable.fasta", ">P1\nPEPTIDEK\n");
        let binary = Scratch::new("unreadable.bin");
        std::fs::write(binary.path(), [0xffu8, 0xfe, 0x00, 0x41, 0x0a, 0xc3]).unwrap();
        let missing = std::env::temp_dir().join("msspeculator-no-such-library.tsv");
        let unparseable = Scratch::holding("unparseable.tsv", "PrecursorMz\n1.0\n");
        let _sidecar = Scratch::holding_at(sidecar_path(unparseable.path()), "{ not json");
        for library in [binary.path(), missing.as_path(), unparseable.path()] {
            assert_eq!(
                check_library(library, None, &options(fasta.path())).unwrap(),
                LibraryCheck::Unknown,
                "{}",
                library.display()
            );
        }
    }

    /// `--config-out` puts the sidecar somewhere else, and then it is still the only copy a TSV
    /// has. A check that looked only beside the library would report every one of those as
    /// carrying no provenance.
    #[test]
    fn a_sidecar_that_was_written_elsewhere_is_still_the_librarys_provenance() {
        let fasta = Scratch::holding("elsewhere.fasta", ">P1\nPEPTIDEK\n");
        let library = tsv("elsewhere.tsv");
        let named = Scratch::new("elsewhere-config.json");
        let mut provenance = crate::provenance::tests::sample();
        provenance.settings =
            Settings::resolve(&options(fasta.path()), BuiltinModel::SmallV0.digest()).unwrap();
        write_sidecar(named.path(), &provenance, &LibraryStats::default()).unwrap();
        assert_eq!(
            check_library(library.path(), None, &options(fasta.path())).unwrap(),
            LibraryCheck::Unknown,
            "nothing beside the library to find"
        );
        assert_eq!(
            check_library(library.path(), Some(named.path()), &options(fasta.path())).unwrap(),
            LibraryCheck::Same
        );
    }

    /// A compressed library carries its header the same way, so the check costs the same open.
    #[test]
    fn a_compressed_library_is_read_through_its_own_suffix() {
        let fasta = Scratch::holding("gz.fasta", ">P1\nPEPTIDEK\n");
        let library = Scratch::new("gz.mzspeclib.txt.gz");
        let mut provenance = crate::provenance::tests::sample();
        provenance.settings =
            Settings::resolve(&options(fasta.path()), BuiltinModel::SmallV0.digest()).unwrap();
        let mut sink = crate::mzspeclib::MzSpecLibSink::new(
            flate2::write::GzEncoder::new(
                std::fs::File::create(library.path()).unwrap(),
                flate2::Compression::default(),
            ),
            library.path(),
        );
        sink.header(&provenance).unwrap();
        sink.finish().unwrap();
        drop(sink);
        assert_eq!(
            check_library(library.path(), None, &options(fasta.path())).unwrap(),
            LibraryCheck::Same
        );
    }
}
