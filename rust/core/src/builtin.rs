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
    SmallV0,
}

impl BuiltinModel {
    /// Stable name used in provenance and by the command-line interface.
    pub const fn name(self) -> &'static str {
        match self {
            Self::SmallV0 => "small-v0",
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
    /// Stable human-readable source used in provenance and logs.
    pub fn spec(&self) -> String {
        match self {
            Self::Builtin(model) => format!("{BUILTIN_PREFIX}{}", model.name()),
            Self::File(path) => path.to_string_lossy().into_owned(),
        }
    }
}

/// Ceiling on one bundled artifact. Enforced by a test, because the cost of a mistake here is a
/// large binary in git history, which is not something a later commit can take back.
pub const MAX_BUNDLED_BYTES: usize = 4 << 20;

/// `(name, bytes, blake2b-256 as recorded when it was vendored)`.
///
/// The digest is checked against the bytes by a test, so replacing the file without stating the
/// new identity fails the build rather than silently shipping different weights under one name.
///
/// `small-v0` is the `small` preset at 0.8054 mean per-dataset spectral agreement, from the
/// plateau-decay run. The `v0` is deliberate: the preset sweep that picks a production model has
/// not run, so this is the plumbing's payload rather than a blessed release.
const BUNDLED: [(&str, &[u8], &str); 1] = [(
    "small-v0",
    include_bytes!("../data/weights/small-v0.safetensors"),
    "8dc9edf567606df1ce2b98530d679ebce139f364021062b72a20d4eaca7162a3",
)];

/// blake2b-256 of some bytes, hex.
///
/// Identity rather than integrity: two libraries generated from the same model and settings are
/// the same library, and a name means nothing without it.
pub fn digest_bytes(bytes: &[u8]) -> String {
    use blake2::digest::{Update, VariableOutput};
    let mut hasher = blake2::Blake2bVar::new(32).expect("32 is a valid blake2b output length");
    hasher.update(bytes);
    let mut out = [0u8; 32];
    hasher.finalize_variable(&mut out).expect("32-byte output");
    out.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Names of every artifact compiled into this build.
pub fn names() -> Vec<&'static str> {
    BUNDLED.iter().map(|(name, _, _)| *name).collect()
}

/// Load one of the artifacts compiled into this crate.
pub fn load_builtin(model: BuiltinModel) -> Result<LoadedModel> {
    let name = model.name();
    let (_, bytes, recorded) = BUNDLED
        .iter()
        .find(|(bundled, _, _)| *bundled == name)
        .ok_or_else(|| anyhow!("builtin model {name:?} is not included in this build"))?;
    Ok(LoadedModel {
        artifact: Artifact::from_bytes(bytes)
            .with_context(|| format!("loading builtin model {name}"))?,
        spec: format!("{BUILTIN_PREFIX}{name}"),
        digest: (*recorded).to_string(),
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
    match spec.strip_prefix(BUILTIN_PREFIX) {
        Some(name) => {
            let (_, bytes, recorded) = BUNDLED
                .iter()
                .find(|(bundled, _, _)| *bundled == name)
                .ok_or_else(|| {
                    anyhow!(
                        "unknown builtin model {name:?}; this build carries: {}",
                        names().join(", ")
                    )
                })?;
            Ok(LoadedModel {
                artifact: Artifact::from_bytes(bytes)
                    .with_context(|| format!("loading builtin model {name}"))?,
                spec: spec.to_string(),
                digest: (*recorded).to_string(),
            })
        }
        None => {
            let bytes = std::fs::read(spec).with_context(|| format!("reading {spec}"))?;
            Ok(LoadedModel {
                artifact: Artifact::from_bytes(&bytes)
                    .with_context(|| format!("loading {spec}"))?,
                spec: spec.to_string(),
                digest: digest_bytes(&bytes),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_bundled_artifact_matches_its_recorded_digest() {
        assert!(
            !BUNDLED.is_empty(),
            "a build with no weights cannot predict"
        );
        for (name, bytes, recorded) in BUNDLED {
            assert_eq!(
                digest_bytes(bytes),
                recorded,
                "{name} bytes do not match the digest recorded for them"
            );
            assert!(
                bytes.len() <= MAX_BUNDLED_BYTES,
                "{name} is {} bytes, over the {MAX_BUNDLED_BYTES} ceiling; weights this large \
                 belong in a runtime fetch, not in git",
                bytes.len()
            );
        }
    }

    #[test]
    fn a_bundled_name_loads_and_reports_itself() {
        let loaded = load_model("builtin:small-v0").unwrap();
        assert_eq!(loaded.spec, "builtin:small-v0");
        assert_eq!(loaded.digest, BUNDLED[0].2);
        // A real trained artifact, not a stub: it carries the norm the corpus was standardized on.
        assert!(loaded.artifact.meta.norm.rt_std > 0.0);
        assert!(names().contains(&"small-v0"));
    }

    #[test]
    fn typed_source_loads_the_builtin_model() {
        let loaded = load_source(ModelSource::Builtin(BuiltinModel::SmallV0)).unwrap();
        assert_eq!(loaded.spec, "builtin:small-v0");
        assert_eq!(loaded.digest, BUNDLED[0].2);
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
        assert_eq!(loaded.digest, BUNDLED[0].2);
    }
}
