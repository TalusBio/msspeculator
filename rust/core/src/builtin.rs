//! Weights compiled into the binary, and the addressing that reaches them.
//!
//! A clean `git clone` has to build a tool that can predict, so the weights are vendored beside
//! `unimod.tsv` and embedded the same way. Downloading them during the build was the alternative
//! and it defeats the requirement: no offline build, a fetch dependency inside `cross`'s
//! container, a host to keep alive, and a cache to invalidate; all to avoid committing half a
//! megabyte.
//!
//! Anything too large for version control does not belong here. It would be a *runtime* fetch
//! into a cache, which leaves this property intact.

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::artifact::Artifact;

/// Prefix that addresses a bundled artifact rather than a filesystem path.
pub const BUILTIN_PREFIX: &str = "builtin:";

/// A model compiled into this crate.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum BuiltinModel {
    /// The `small` preset at 0.8054 mean per-dataset spectral agreement, from the plateau-decay
    /// run. The `v0` is deliberate: the preset sweep that picks a production model has not run,
    /// so this is the plumbing's payload rather than a blessed release.
    SmallV0,
}

impl BuiltinModel {
    /// Every variant, so a lookup or a listing cannot go stale against the enum.
    const ALL: [Self; 1] = [Self::SmallV0];

    /// Stable name used in provenance and by the command-line interface.
    pub const fn name(self) -> &'static str {
        match self {
            Self::SmallV0 => "small-v0",
        }
    }

    /// The variant a name addresses, or `None` for a name this build does not carry.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|model| model.name() == name)
    }

    /// The weights compiled into this build.
    ///
    /// Reaching them is infallible because the enum is closed and every variant names its own
    /// payload here. A lookup by name would have to answer "what if it is missing", which for a
    /// compiled-in artifact is not a question the caller can do anything about.
    pub const fn bytes(self) -> &'static [u8] {
        match self {
            Self::SmallV0 => include_bytes!("../data/weights/small-v0.safetensors"),
        }
    }

    /// The digest recorded for these weights when they were vendored, without loading them.
    ///
    /// The same value [`load_builtin`] reports, so a caller that needs only the identity of a
    /// model does not pay for its tensors. `every_bundled_artifact_matches_its_recorded_digest`
    /// checks it against the bytes above, so replacing the file without restating the identity
    /// fails the build rather than silently shipping different weights under one name.
    pub const fn digest(self) -> &'static str {
        match self {
            Self::SmallV0 => "8dc9edf567606df1ce2b98530d679ebce139f364021062b72a20d4eaca7162a3",
        }
    }
}

/// Where an inference artifact should be loaded from.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ModelSource {
    Builtin(BuiltinModel),
    File(PathBuf),
}

impl ModelSource {
    /// `builtin:<name>` addresses the binary; anything else is a filesystem path.
    ///
    /// The only place a `--model` string is interpreted, so the command line, [`load_model`] and
    /// a library's provenance cannot disagree about what one means.
    pub fn from_spec(spec: &str) -> Result<Self> {
        match spec.strip_prefix(BUILTIN_PREFIX) {
            Some(name) => BuiltinModel::from_name(name)
                .map(Self::Builtin)
                .ok_or_else(|| {
                    anyhow!(
                        "unknown builtin model {name:?}; this build carries: {}",
                        names().join(", ")
                    )
                }),
            None => Ok(Self::File(PathBuf::from(spec))),
        }
    }

    /// Stable human-readable source used in provenance and logs.
    pub fn spec(&self) -> String {
        match self {
            Self::Builtin(model) => format!("{BUILTIN_PREFIX}{}", model.name()),
            Self::File(path) => path.to_string_lossy().into_owned(),
        }
    }

    /// The identity of the weights this addresses, without loading them.
    ///
    /// Beside [`spec`](Self::spec) because the two answer one question — which model is this —
    /// and a caller should not have to know that a bundled artifact's digest is a recorded
    /// constant while a file's is a full read of it.
    pub fn digest(&self) -> Result<String> {
        match self {
            Self::Builtin(model) => Ok(model.digest().to_string()),
            Self::File(path) => digest_file(path),
        }
    }
}

/// Ceiling on one bundled artifact. Enforced by a test, because the cost of a mistake here is a
/// large binary in git history, which is not something a later commit can take back.
pub const MAX_BUNDLED_BYTES: usize = 4 << 20;

/// A blake2b-256 hasher and the hex its output is spelled as.
///
/// Identity rather than integrity: two libraries generated from the same model and settings are
/// the same library, and a name means nothing without it.
fn hasher() -> blake2::Blake2bVar {
    use blake2::digest::VariableOutput;
    blake2::Blake2bVar::new(32).expect("32 is a valid blake2b output length")
}

fn hex(hasher: blake2::Blake2bVar) -> String {
    use blake2::digest::VariableOutput;
    let mut out = [0u8; 32];
    hasher.finalize_variable(&mut out).expect("32-byte output");
    out.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// blake2b-256 of some bytes, hex.
pub fn digest_bytes(bytes: &[u8]) -> String {
    use blake2::digest::Update;
    let mut hasher = hasher();
    hasher.update(bytes);
    hex(hasher)
}

/// blake2b-256 of a file's bytes, hex, streamed so a multi-GB input is not held in memory.
pub fn digest_file(path: &std::path::Path) -> Result<String> {
    use blake2::digest::Update;
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).with_context(|| format!("hashing {}", path.display()))?;
    let mut hasher = hasher();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex(hasher))
}

/// Names of every artifact compiled into this build.
pub fn names() -> Vec<&'static str> {
    BuiltinModel::ALL.iter().map(|model| model.name()).collect()
}

/// Load one of the artifacts compiled into this crate.
pub fn load_builtin(model: BuiltinModel) -> Result<LoadedModel> {
    Ok(LoadedModel {
        artifact: Artifact::from_bytes(model.bytes())
            .with_context(|| format!("loading builtin model {}", model.name()))?,
        spec: ModelSource::Builtin(model).spec(),
        digest: model.digest().to_string(),
    })
}

/// Load a model from a typed source.
pub fn load_source(source: ModelSource) -> Result<LoadedModel> {
    match source {
        ModelSource::Builtin(model) => load_builtin(model),
        ModelSource::File(path) => {
            let spec = path.to_string_lossy().into_owned();
            let bytes = std::fs::read(&path).with_context(|| format!("reading {spec}"))?;
            Ok(LoadedModel {
                artifact: Artifact::from_bytes(&bytes)
                    .with_context(|| format!("loading {spec}"))?,
                spec,
                digest: digest_bytes(&bytes),
            })
        }
    }
}

/// An artifact plus how to refer to the thing it came from.
///
/// `Debug` omits the artifact deliberately: printing it would dump every tensor.
pub struct LoadedModel {
    pub artifact: Artifact,
    /// Exactly what the caller asked for: `builtin:small-v0`, or the path as given. A bundled name
    /// is what a library's provenance should record, because a temp path is not an identity.
    pub spec: String,
    pub digest: String,
}

impl std::fmt::Debug for LoadedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedModel")
            .field("spec", &self.spec)
            .field("digest", &self.digest)
            .finish_non_exhaustive()
    }
}

/// Load `builtin:<name>` from the binary, or anything else as a filesystem path.
pub fn load_model(spec: &str) -> Result<LoadedModel> {
    load_source(ModelSource::from_spec(spec)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_bundled_artifact_matches_its_recorded_digest() {
        assert!(
            !BuiltinModel::ALL.is_empty(),
            "a build with no weights cannot predict"
        );
        for model in BuiltinModel::ALL {
            let name = model.name();
            assert_eq!(
                digest_bytes(model.bytes()),
                model.digest(),
                "{name} bytes do not match the digest recorded for them"
            );
            assert!(
                model.bytes().len() <= MAX_BUNDLED_BYTES,
                "{name} is {} bytes, over the {MAX_BUNDLED_BYTES} ceiling; weights this large \
                 belong in a runtime fetch, not in git",
                model.bytes().len()
            );
            assert_eq!(BuiltinModel::from_name(name), Some(model));
        }
        assert_eq!(BuiltinModel::from_name("nonexistent"), None);
    }

    #[test]
    fn a_source_reports_its_digest_without_loading_the_tensors() {
        assert_eq!(
            ModelSource::Builtin(BuiltinModel::SmallV0)
                .digest()
                .unwrap(),
            BuiltinModel::SmallV0.digest()
        );
        // The same identity the loader reports, reached without paying for the artifact.
        assert_eq!(
            ModelSource::Builtin(BuiltinModel::SmallV0)
                .digest()
                .unwrap(),
            load_builtin(BuiltinModel::SmallV0).unwrap().digest
        );
        // Streaming a file in chunks and hashing it whole are two paths to one identity: a check
        // reads the model with `digest_file` while the build hashes the bytes it loaded, and a
        // library compares equal to itself only if those agree.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/weights/small-v0.safetensors"
        );
        assert_eq!(
            digest_file(std::path::Path::new(path)).unwrap(),
            digest_bytes(&std::fs::read(path).unwrap())
        );
    }

    #[test]
    fn a_bundled_name_loads_and_reports_itself() {
        let loaded = load_model("builtin:small-v0").unwrap();
        assert_eq!(loaded.spec, "builtin:small-v0");
        assert_eq!(loaded.digest, BuiltinModel::SmallV0.digest());
        // A real trained artifact, not a stub: it carries the norm the corpus was standardized on.
        assert!(loaded.artifact.meta.norm.rt_std > 0.0);
        assert!(names().contains(&"small-v0"));
    }

    #[test]
    fn typed_source_loads_the_builtin_model() {
        let loaded = load_source(ModelSource::Builtin(BuiltinModel::SmallV0)).unwrap();
        assert_eq!(loaded.spec, "builtin:small-v0");
        assert_eq!(loaded.digest, BuiltinModel::SmallV0.digest());
    }

    #[test]
    fn an_unknown_builtin_names_what_the_build_has() {
        let error = load_model("builtin:nonexistent").unwrap_err().to_string();
        assert!(
            error.contains("nonexistent") && error.contains("small-v0"),
            "{error}"
        );
    }

    #[test]
    fn a_path_still_loads_and_is_hashed_from_its_bytes() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/weights/small-v0.safetensors"
        );
        let loaded = load_model(path).unwrap();
        assert_eq!(loaded.spec, path);
        // Same bytes reached two ways must have one identity.
        assert_eq!(loaded.digest, BuiltinModel::SmallV0.digest());
    }
}
