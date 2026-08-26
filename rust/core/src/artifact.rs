//! Load a pepdistill `.safetensors` artifact: f32 tensors by name + the JSON metadata blob.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use safetensors::SafeTensors;
use serde::Deserialize;

/// Artifact schema this build reads.
///
/// v2 replaced the single scaled `mod_proj` scalar with the two-encoder (`comp_enc` /
/// `mass_enc`) modification representation and made the N/C-term tokens mandatory, so a v1
/// artifact's tensors mean something different.
///
/// v3 added the ChromRunbook's per-dataset RT output affine (`runbook.log_scale.weight`,
/// `runbook.shift.weight`). A v2 artifact simply lacks those tensors. Treating their absence
/// as identity would be a plausible guess, but it is still a guess about what a file means.
/// A v2 checkpoint cannot load into the current runbook anyway. Reject and re-export.
///
/// Named acquisition setups (`enc.setup_emb.weight` + `ms_context_index`) were added WITHOUT a
/// bump, and the difference from v3 is the whole rule: a missing affine left the reader
/// guessing what a dataset row meant, while a missing setup index says something exact. No
/// setup was ever named. Since the version check is strict equality, bumping would force every
/// published artifact to be re-exported to buy nothing.
pub const FORMAT_VERSION: u32 = 3;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub backbone: String,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ff_mult: usize,
    pub activation: String,
    pub max_len: usize,
    pub max_charge: usize,
    pub n_ion: usize,
    pub context_dim: usize,
}

#[derive(Debug, Deserialize)]
pub struct Norm {
    pub rt_mean: f32,
    pub rt_std: f32,
    pub ccs_mean: f32,
    pub ccs_std: f32,
}

#[derive(Debug, Deserialize)]
pub struct Vocab {
    pub instruments: Vec<String>,
    pub detectors: Vec<String>,
    pub fragmentations: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct Meta {
    pub format_version: u32,
    pub config: Config,
    pub norm: Norm,
    #[serde(default)]
    pub has_encoder: bool,
    #[serde(default)]
    pub has_runbook: bool,
    #[serde(default)]
    pub vocab: Option<Vocab>,
    #[serde(default)]
    pub dataset_index: Option<HashMap<String, i64>>,
    /// Setup name -> `enc.setup_emb.weight` row, for a context fitted against a source that
    /// records no acquisition factors. Absent means none was ever named.
    #[serde(default)]
    pub ms_context_index: Option<HashMap<String, i64>>,
}

/// A loaded artifact: raw f32 tensors keyed by name + parsed metadata.
pub struct Artifact {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
    /// The metadata blob exactly as the file carried it. Kept beside the parsed `meta` so that
    /// writing an artifact back preserves keys this build does not model, instead of dropping
    /// them through a round trip of the structs above.
    raw_meta: serde_json::Value,
    pub meta: Meta,
}

/// One f32 tensor, borrowed, for the safetensors writer.
struct F32View<'a> {
    shape: &'a [usize],
    data: &'a [f32],
}

impl safetensors::View for F32View<'_> {
    fn dtype(&self) -> safetensors::Dtype {
        safetensors::Dtype::F32
    }

    fn shape(&self) -> &[usize] {
        self.shape
    }

    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        let mut bytes = Vec::with_capacity(self.data.len() * 4);
        for value in self.data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::borrow::Cow::Owned(bytes)
    }

    fn data_len(&self) -> usize {
        self.data.len() * 4
    }
}

fn le_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    let (words, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(anyhow!(
            "f32 tensor has {} bytes, which is not divisible by 4",
            bytes.len()
        ));
    }
    Ok(words.iter().map(|b| f32::from_le_bytes(*b)).collect())
}

impl Artifact {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let buf = std::fs::read(path.as_ref())
            .with_context(|| format!("reading {}", path.as_ref().display()))?;
        Self::from_bytes(&buf)
    }

    /// Read an artifact already in memory, which is how a bundled one arrives: `include_bytes!`
    /// hands over a `&'static [u8]` that was never a file on the machine doing the reading.
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        let (_, header) = SafeTensors::read_metadata(buf).context("parsing safetensors header")?;
        let map = header
            .metadata()
            .as_ref()
            .ok_or_else(|| anyhow!("artifact has no __metadata__"))?;
        let blob = map
            .get("pepdistill")
            .ok_or_else(|| anyhow!("__metadata__ missing 'pepdistill' key"))?;
        let raw_meta: serde_json::Value =
            serde_json::from_str(blob).context("parsing pepdistill metadata JSON")?;
        let meta: Meta =
            serde_json::from_value(raw_meta.clone()).context("reading pepdistill metadata")?;
        if meta.format_version != FORMAT_VERSION {
            return Err(anyhow!(
                "artifact format_version {} is not supported (this build reads {}); \
                 v1 predates the composition mod encoding and v2 predates the per-dataset RT \
                 affine, so their tensors do not mean what this build expects. Re-export from \
                 a retrained checkpoint",
                meta.format_version,
                FORMAT_VERSION
            ));
        }
        if meta.config.backbone != "transformer" {
            return Err(anyhow!(
                "only the transformer backbone is supported (got {:?})",
                meta.config.backbone
            ));
        }

        let st = SafeTensors::deserialize(buf).context("deserializing tensors")?;
        let mut tensors = HashMap::new();
        for name in st.names() {
            let view = st.tensor(name)?;
            if view.dtype() != safetensors::Dtype::F32 {
                return Err(anyhow!("tensor {name} is not f32"));
            }
            let data = le_f32(view.data()).with_context(|| format!("decoding tensor {name}"))?;
            tensors.insert(name.clone(), (view.shape().to_vec(), data));
        }
        Ok(Self {
            tensors,
            raw_meta,
            meta,
        })
    }

    /// Write this artifact, tensors and metadata, to `path`.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let views: Vec<(&String, F32View<'_>)> = self
            .tensors
            .iter()
            .map(|(name, (shape, data))| (name, F32View { shape, data }))
            .collect();
        let info = HashMap::from([("pepdistill".to_string(), self.raw_meta.to_string())]);
        safetensors::serialize_to_file(views, &Some(info), path.as_ref())
            .with_context(|| format!("writing {}", path.as_ref().display()))
    }

    /// Store `context` as the named acquisition setup `name` and return the row it occupies.
    ///
    /// A name already present is overwritten in place, so re-fitting a setup cannot leave the
    /// artifact holding two rows under one name. A new name appends, so a row already fitted
    /// keeps both its index and its weights.
    pub fn set_ms_context(&mut self, name: &str, context: &[f32]) -> Result<i64> {
        let mut index = self.meta.ms_context_index.clone().unwrap_or_default();
        // Row 0 is the unnamed setup and must stay zero: it is what an unnamed source gets, and
        // the Python encoder pins it with `padding_idx=0`.
        let rows = match self.tensors.get("enc.setup_emb.weight") {
            Some((shape, _)) if shape.len() != 2 => {
                return Err(anyhow!("setup table is {shape:?}, expected 2-D"))
            }
            Some((shape, _)) if shape[1] != context.len() => {
                return Err(anyhow!(
                    "setup table holds {}-d contexts, got {}",
                    shape[1],
                    context.len()
                ))
            }
            Some((shape, _)) => shape[0],
            None => 1,
        };
        let row = match index.get(name) {
            Some(row) => *row as usize,
            None => rows,
        };
        let (shape, data) = self
            .tensors
            .entry("enc.setup_emb.weight".to_string())
            .or_insert_with(|| (vec![1, context.len()], vec![0.0; context.len()]));
        if row >= shape[0] {
            data.resize((row + 1) * context.len(), 0.0);
            shape[0] = row + 1;
        }
        data[row * context.len()..(row + 1) * context.len()].copy_from_slice(context);
        index.insert(name.to_string(), row as i64);
        self.raw_meta["ms_context_index"] = serde_json::to_value(&index)?;
        self.meta.ms_context_index = Some(index);
        Ok(row as i64)
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn raw(&self, name: &str) -> Result<&(Vec<usize>, Vec<f32>)> {
        self.tensors
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))
    }

    pub fn get1(&self, name: &str) -> Result<ndarray::ArrayView1<'_, f32>> {
        let (shape, data) = self.raw(name)?;
        if shape.len() != 1 {
            return Err(anyhow!("tensor {name} is {shape:?}, expected 1-D"));
        }
        Ok(ndarray::ArrayView1::from(data.as_slice()))
    }

    pub fn get2(&self, name: &str) -> Result<ndarray::ArrayView2<'_, f32>> {
        let (shape, data) = self.raw(name)?;
        if shape.len() != 2 {
            return Err(anyhow!("tensor {name} is {shape:?}, expected 2-D"));
        }
        ndarray::ArrayView2::from_shape((shape[0], shape[1]), data.as_slice())
            .with_context(|| format!("reshaping {name}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_f32_bytes_are_refused() {
        let err = le_f32(&[0, 0, 0, 0, 1]).unwrap_err();
        assert!(err.to_string().contains("not divisible by 4"), "{err}");
    }

    /// The smallest file `load` accepts: the version and backbone gates, and one tensor so the
    /// file is not empty. Named-setup behaviour is about the metadata and one table, so the rest
    /// of a real artifact would only be noise here.
    fn minimal(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("pepdistill-{name}.safetensors"));
        let meta = serde_json::json!({
            "format_version": FORMAT_VERSION,
            "config": {"backbone": "transformer", "d_model": 4, "n_layers": 1, "n_heads": 1,
                       "ff_mult": 2, "activation": "gelu", "max_len": 8, "max_charge": 4,
                       "n_ion": 4, "context_dim": 3},
            "norm": {"rt_mean": 0.0, "rt_std": 1.0, "ccs_mean": 0.0, "ccs_std": 1.0},
            // A key this build does not model, to prove a write does not drop it.
            "provenance": "test",
        });
        let info = HashMap::from([("pepdistill".to_string(), meta.to_string())]);
        let data = vec![0.0f32; 4];
        let view = F32View {
            shape: &[4],
            data: &data,
        };
        safetensors::serialize_to_file([("model.rt_head.0.bias", view)], &Some(info), &path)
            .expect("writing the fixture");
        path
    }

    #[test]
    fn a_named_setup_survives_a_write_and_read() {
        let path = minimal("named-setup");
        let mut art = Artifact::load(&path).expect("loading the fixture");
        assert_eq!(
            art.set_ms_context("Evosep60SPD_heron", &[1.0, 2.0, 3.0])
                .unwrap(),
            1
        );

        let out = std::env::temp_dir().join("pepdistill-named-setup-out.safetensors");
        art.save(&out).expect("writing");
        let reloaded = Artifact::load(&out).expect("reloading");

        let index = reloaded.meta.ms_context_index.as_ref().expect("index");
        assert_eq!(index["Evosep60SPD_heron"], 1);
        let table = reloaded.get2("enc.setup_emb.weight").unwrap();
        assert_eq!(table.shape(), [2, 3]);
        // Row 0 stays the unnamed setup, and a metadata key this build ignores stays put.
        assert_eq!(table.row(0).to_vec(), vec![0.0, 0.0, 0.0]);
        assert_eq!(table.row(1).to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(reloaded.raw_meta["provenance"], "test");
    }

    #[test]
    fn refitting_a_setup_replaces_its_row_rather_than_adding_one() {
        let path = minimal("refit-setup");
        let mut art = Artifact::load(&path).unwrap();
        art.set_ms_context("heron", &[1.0, 1.0, 1.0]).unwrap();
        art.set_ms_context("cyspat", &[2.0, 2.0, 2.0]).unwrap();

        assert_eq!(art.set_ms_context("heron", &[9.0, 9.0, 9.0]).unwrap(), 1);
        let table = art.get2("enc.setup_emb.weight").unwrap();
        assert_eq!(table.shape(), [3, 3]);
        assert_eq!(table.row(1).to_vec(), vec![9.0, 9.0, 9.0]);
        assert_eq!(table.row(2).to_vec(), vec![2.0, 2.0, 2.0]); // untouched by the refit
    }

    #[test]
    fn a_context_of_the_wrong_width_is_refused() {
        let path = minimal("wrong-width");
        let mut art = Artifact::load(&path).unwrap();
        art.set_ms_context("heron", &[1.0, 1.0, 1.0]).unwrap();
        let err = art.set_ms_context("cyspat", &[1.0, 1.0]).unwrap_err();
        assert!(err.to_string().contains("3-d contexts"), "{err}");
    }
}
