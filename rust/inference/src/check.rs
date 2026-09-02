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

/// The blocks of a provenance that only a build can fill, by top-level key.
///
/// Which release wrote the library, where it went, and how this artifact's retention anchors
/// measured. Two runs that would generate the same library disagree about all three, so
/// comparing them would report every rebuild as different.
const NOT_A_SETTING: [&str; 3] = ["generator", "output", "retention"];

/// The two settings keys a check ignores.
///
/// They sit in `inputs` beside the digests, but they are paths as they were typed.
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
    /// Nothing to compare against: no `msspeculator:` attributes in the header, and no provenance
    /// of ours at either sidecar path. Another tool's library, one of ours written with
    /// `--no-config-out` in a format that carries no header, or a file this could not read at
    /// all. Not a mismatch.
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
    ///
    /// Says which side each value came from rather than which came first. A rebuild is replacing
    /// the library and could say "was, now"; a search comparing a library it did not write is not
    /// replacing anything, and there is one type for both.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let side = |value: &Option<String>| value.clone().unwrap_or_else(|| "unset".to_string());
        write!(
            f,
            "{}: library {}, expected {}",
            self.key,
            side(&self.library),
            side(&self.expected)
        )
    }
}

/// Whether the library at `library` was built the way `opts` describes.
///
/// Resolves what `opts` mean and hands them to [`check_against`]. That resolution is the whole
/// cost: it hashes the FASTA, which is a full read of it, and it hashes a file-backed model. A
/// caller that already holds the [`Settings`] it is about to write should call `check_against`
/// and pay neither.
///
/// The options are not validated here. A check is read-only, and refusing to compare because a
/// charge range is inverted would report an options-side mistake as something wrong with the
/// library on disk.
pub fn check_library(
    library: &Path,
    sidecar: Option<&Path>,
    opts: &StreamOptions<'_>,
) -> Result<LibraryCheck> {
    let expected = Settings::resolve(opts, &opts.model.digest()?)?;
    Ok(check_against(library, sidecar, &expected))
}

/// Whether the library at `library` was built with `expected`.
///
/// For a search reading someone else's library: warn per [`Difference`], which prints both
/// values, and run anyway. For a build about to overwrite one: [`LibraryCheck::Same`] means the
/// file already is what this run would produce.
///
/// Reads the header and stops at the end of it, so the library costs one open whatever its size.
/// A format with no header to carry provenance is compared against its sidecar instead: `sidecar`
/// is tried first when given, then the conventional [`sidecar_path`] beside the library. Both,
/// because `--config-out` moves where a sidecar is *written* without moving the one an earlier
/// build already left beside the library, and a check that guessed either way alone would report
/// half of those as carrying no provenance.
///
/// Only what the settings determine is compared: see [`Settings`] for what that is and is not.
/// Infallible by construction. Every way a library can fail to answer the question, including
/// being unreadable, is [`LibraryCheck::Unknown`], because a check that raises is a check that
/// can fail a build it was only advising.
pub fn check_against(library: &Path, sidecar: Option<&Path>, expected: &Settings) -> LibraryCheck {
    // The header first, because that copy cannot be separated from the library it describes. A
    // DIA-NN TSV has no header to put it in, so for that format the sidecar is the only copy.
    let Some(recorded) = from_header(library)
        .or_else(|| sidecar.and_then(from_sidecar))
        .or_else(|| from_sidecar(&sidecar_path(library)))
    else {
        return LibraryCheck::Unknown;
    };
    compare(comparable(recorded), comparable(expected.attributes()))
}

/// The pairs the library's own header carries, or `None` when it carries none of ours.
///
/// Anything at all is ours: [`crate::mzspeclib::header_attributes`] keeps only the
/// `msspeculator:` pairs, so the prefix has already done the identifying.
fn from_header(library: &Path) -> Option<BTreeMap<String, String>> {
    let pairs = crate::library::open_library(library)
        .map(crate::mzspeclib::header_attributes)
        .unwrap_or_default();
    (!pairs.is_empty()).then_some(pairs)
}

/// The provenance beside a library, in the spelling a header would have carried it.
///
/// Nothing here is an error: a missing sidecar is the ordinary case, and a file at that name that
/// is not one of ours says nothing about the library either way. Both come back as no provenance,
/// because a check that refuses to run is worse than one that reports it has nothing to compare.
///
/// A `generator` block is what says a document is one of ours. Unlike a header, a JSON file has
/// no prefix on its keys, and without that test any parseable JSON sitting at the sidecar's name
/// would narrow to nothing and report every expected key as newly set.
///
/// In the header's spelling deliberately: a library's two copies of its provenance are the same
/// provenance, so whichever one a reader finds has to compare the same way. The sidecar keeps the
/// nulls a header drops, and those fall out here.
fn from_sidecar(path: &Path) -> Option<BTreeMap<String, String>> {
    let text = std::fs::read_to_string(path).ok()?;
    let config = serde_json::from_str(&text).ok()?;
    let pairs = crate::provenance::flatten(&config);
    pairs.contains_key("generator.tool").then_some(pairs)
}

/// A provenance narrowed to the keys two builds can disagree about.
///
/// A denylist, and that direction is the point. Narrowing the library side down to the keys
/// *this* build knows would drop a settings group that a later release added, and then answer
/// `Same` about a library built with a knob this binary cannot honour. Everything a build fills
/// in on its own is named in [`NOT_A_SETTING`]; anything else is a setting, including one this
/// build has never heard of, which arrives as a [`Difference`] with no expected side.
///
/// `narrowing_the_document_leaves_exactly_the_settings` holds the two lists against the struct.
fn comparable(pairs: BTreeMap<String, String>) -> BTreeMap<String, String> {
    pairs
        .into_iter()
        .filter(|(key, _)| {
            let prefix = key.split_once('.').map_or(key.as_str(), |(head, _)| head);
            !NOT_A_SETTING.contains(&prefix) && !NOT_AN_IDENTITY.contains(&key.as_str())
        })
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
    use std::path::PathBuf;

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

    /// The provenance a real build writes for these options.
    ///
    /// An output block and an anchor check measured from a loaded artifact ride along, neither of
    /// which a reader checking the library can have. Both are therefore what the comparison has to
    /// look past, so both are here rather than defaulted away.
    fn provenance_for(opts: &StreamOptions<'_>) -> crate::provenance::LibraryProvenance {
        let mut provenance = crate::provenance::tests::sample();
        provenance.settings = Settings::resolve(opts, BuiltinModel::SmallV0.digest()).unwrap();
        provenance
    }

    /// A library whose provenance lives only in a sidecar, which is the DIA-NN TSV case.
    fn built_at(sidecar: PathBuf, opts: &StreamOptions<'_>) -> Scratch {
        let sidecar = Scratch::at(sidecar);
        write_sidecar(
            sidecar.path(),
            &provenance_for(opts),
            &LibraryStats::default(),
        )
        .unwrap();
        sidecar
    }

    /// The same, beside the library it describes, which is where a check looks by default.
    fn built(library: &Scratch, opts: &StreamOptions<'_>) -> Scratch {
        built_at(sidecar_path(library.path()), opts)
    }

    /// A library whose provenance is in its own header, which is the mzSpecLib case.
    fn built_with_header(library: &Scratch, opts: &StreamOptions<'_>, compress: bool) {
        let file = std::fs::File::create(library.path()).unwrap();
        let writer: Box<dyn std::io::Write + Send> = if compress {
            Box::new(flate2::write::GzEncoder::new(
                file,
                flate2::Compression::default(),
            ))
        } else {
            Box::new(file)
        };
        let mut sink = crate::mzspeclib::MzSpecLibSink::new(writer, library.path());
        sink.header(&provenance_for(opts)).unwrap();
        sink.finish().unwrap();
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
    /// Adding a `StreamOptions` field means adding a line here.
    #[test]
    fn every_option_that_changes_the_library_changes_the_settings() {
        let fasta = Scratch::holding("knobs.fasta", ">P1\nPEPTIDEKPEPTIDER\n");
        let other = Scratch::holding("knobs-other.fasta", ">P1\nPEPTIDEKPEPTIDERK\n");
        let weights = Scratch::holding("knobs.safetensors", "not a real artifact, only hashed");
        let baseline = settings(&options(fasta.path()));
        let named = msspeculator_core::MsContext::Named("setup".into());
        let mods = ["C[UNIMOD:4]".to_string()];
        // The lifetime is named rather than elided: written as `&dyn Fn(&mut StreamOptions<'_>)`
        // the struct's lifetime is higher-ranked, and every borrowed field a knob assigns would
        // have to be `'static`.
        type Knob<'a> = dyn Fn(&mut StreamOptions<'a>) + 'a;
        let knobs: [(&str, &Knob<'_>); 16] = [
            ("model", &|o| {
                o.model = ModelSource::File(weights.path().into());
            }),
            ("fasta", &|o| o.fasta = other.path()),
            ("activation", &|o| o.activation = Some("relu")),
            ("ms_context", &|o| o.ms_context = Some(&named)),
            ("chrom_context", &|o| o.chrom_context = Some("dsA")),
            ("min_intensity", &|o| o.min_intensity = 0.5),
            ("missed_cleavages", &|o| o.missed_cleavages = 3),
            ("min_length", &|o| o.min_length = 8),
            ("max_length", &|o| o.max_length = 25),
            ("min_charge", &|o| o.min_charge = 3),
            ("max_charge", &|o| o.max_charge = 5),
            ("fixed_mods", &|o| o.fixed_mods = &mods),
            ("variable_mods", &|o| o.variable_mods = &mods),
            ("max_variable_mods", &|o| o.max_variable_mods = 2),
            ("max_fragments", &|o| o.max_fragments = Some(8)),
            ("generate_decoys", &|o| o.generate_decoys = true),
        ];
        for (knob, change) in knobs {
            let mut changed = options(fasta.path());
            change(&mut changed);
            assert_ne!(
                settings(&changed),
                baseline,
                "{knob} changes the library but not what a check compares"
            );
        }
        // `progress` is the one option that does not, because it changes nothing on the way out.
        let mut watched = options(fasta.path());
        watched.progress = Some(&|_| {});
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
            "fragments.max_fragments: library 15, expected unset"
        );
    }

    /// The header is the copy that cannot be separated from the library, so it wins when both
    /// exist — and it has to compare the same way the sidecar does, nulls and software version
    /// included.
    #[test]
    fn a_header_is_read_in_place_of_a_sidecar_and_agrees_with_one() {
        let fasta = Scratch::holding("header.fasta", ">P1\nPEPTIDEK\n");
        let library = Scratch::new("header.mzspeclib.txt");
        built_with_header(&library, &options(fasta.path()), false);
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
        let _sidecar = built_at(named.path().to_path_buf(), &options(fasta.path()));
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

    /// A named sidecar is where a check looks *first*, not the only place it looks. `--config-out`
    /// moves where the next sidecar is written; it says nothing about where an earlier build left
    /// the one describing the file being replaced.
    #[test]
    fn the_conventional_sidecar_is_still_read_when_another_is_named() {
        let fasta = Scratch::holding("both.fasta", ">P1\nPEPTIDEK\n");
        let library = tsv("both.tsv");
        let _beside_it = built(&library, &options(fasta.path()));
        let not_written_yet = Scratch::new("both-elsewhere.json").path().to_path_buf();
        assert_eq!(
            check_library(
                library.path(),
                Some(&not_written_yet),
                &options(fasta.path())
            )
            .unwrap(),
            LibraryCheck::Same
        );
    }

    /// A compressed library carries its header the same way, so the check costs the same open.
    #[test]
    fn a_compressed_library_is_read_through_its_own_suffix() {
        let fasta = Scratch::holding("gz.fasta", ">P1\nPEPTIDEK\n");
        let library = Scratch::new("gz.mzspeclib.txt.gz");
        built_with_header(&library, &options(fasta.path()), true);
        assert_eq!(
            check_library(library.path(), None, &options(fasta.path())).unwrap(),
            LibraryCheck::Same
        );
    }

    /// [`NOT_A_SETTING`] and [`NOT_AN_IDENTITY`] are the complement of [`Settings`] within the
    /// published document, and nothing in the type system says so. Narrowing the whole document
    /// has to leave exactly the narrowed settings: a build-only block added to
    /// [`crate::provenance::LibraryProvenance`] without being listed would start being compared
    /// and report every rebuild as different, and a settings key wrongly listed would stop being
    /// compared and report two different libraries as the same.
    #[test]
    fn narrowing_the_document_leaves_exactly_the_settings() {
        let provenance = crate::provenance::tests::sample();
        assert_eq!(
            comparable(crate::provenance::flatten(&provenance.to_json())),
            comparable(provenance.settings.attributes())
        );
    }

    /// The keys this build has never heard of are the ones worth reporting: a library built by a
    /// later release records settings under groups that do not exist here yet, and answering
    /// `Same` about one would claim a knob this binary cannot honour made no difference.
    #[test]
    fn a_setting_group_from_a_later_release_is_a_difference() {
        let fasta = Scratch::holding("future.fasta", ">P1\nPEPTIDEK\n");
        let library = tsv("future.tsv");
        let sidecar = built(&library, &options(fasta.path()));
        let mut document: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(sidecar.path()).unwrap()).unwrap();
        document["enzymes"] = serde_json::json!({ "name": "chymotrypsin" });
        std::fs::write(sidecar.path(), document.to_string()).unwrap();
        assert_eq!(
            check_library(library.path(), None, &options(fasta.path())).unwrap(),
            LibraryCheck::Different(vec![Difference {
                key: "enzymes.name".into(),
                library: Some("chymotrypsin".into()),
                expected: None,
            }])
        );
    }

    /// A JSON file that happens to sit at the sidecar's name is not a provenance, and reporting
    /// its keys as settings would answer a question about the library from a document that says
    /// nothing about it.
    #[test]
    fn json_that_is_not_a_provenance_is_unknown() {
        let fasta = Scratch::holding("stranger.fasta", ">P1\nPEPTIDEK\n");
        let library = tsv("stranger.tsv");
        let _sidecar = Scratch::holding_at(
            sidecar_path(library.path()),
            r#"{"hello": "world", "digestion": {"missed_cleavages": 9}}"#,
        );
        assert_eq!(
            check_library(library.path(), None, &options(fasta.path())).unwrap(),
            LibraryCheck::Unknown
        );
    }
}
