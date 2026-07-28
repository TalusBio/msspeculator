//! Load a pepdistill `.safetensors` artifact: f32 tensors by name + the JSON metadata blob.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use ndarray::{Array1, Array2};
use safetensors::SafeTensors;
use serde::Deserialize;

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
}

/// A loaded artifact: raw f32 tensors keyed by name + parsed metadata.
pub struct Artifact {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
    pub meta: Meta,
}

fn le_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

impl Artifact {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let buf = std::fs::read(path.as_ref())
            .with_context(|| format!("reading {}", path.as_ref().display()))?;

        let (_, header) = SafeTensors::read_metadata(&buf).context("parsing safetensors header")?;
        let map = header
            .metadata()
            .as_ref()
            .ok_or_else(|| anyhow!("artifact has no __metadata__"))?;
        let blob = map
            .get("pepdistill")
            .ok_or_else(|| anyhow!("__metadata__ missing 'pepdistill' key"))?;
        let meta: Meta = serde_json::from_str(blob).context("parsing pepdistill metadata JSON")?;
        if meta.format_version != 1 {
            return Err(anyhow!(
                "unsupported format_version {}",
                meta.format_version
            ));
        }
        if meta.config.backbone != "transformer" {
            return Err(anyhow!(
                "only the transformer backbone is supported (got {:?})",
                meta.config.backbone
            ));
        }

        let st = SafeTensors::deserialize(&buf).context("deserializing tensors")?;
        let mut tensors = HashMap::new();
        for name in st.names() {
            let view = st.tensor(name)?;
            if view.dtype() != safetensors::Dtype::F32 {
                return Err(anyhow!("tensor {name} is not f32"));
            }
            tensors.insert(name.clone(), (view.shape().to_vec(), le_f32(view.data())));
        }
        Ok(Self { tensors, meta })
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn raw(&self, name: &str) -> Result<&(Vec<usize>, Vec<f32>)> {
        self.tensors
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))
    }

    pub fn get1(&self, name: &str) -> Result<Array1<f32>> {
        let (shape, data) = self.raw(name)?;
        if shape.len() != 1 {
            return Err(anyhow!("tensor {name} is {shape:?}, expected 1-D"));
        }
        Ok(Array1::from(data.clone()))
    }

    pub fn get2(&self, name: &str) -> Result<Array2<f32>> {
        let (shape, data) = self.raw(name)?;
        if shape.len() != 2 {
            return Err(anyhow!("tensor {name} is {shape:?}, expected 2-D"));
        }
        Array2::from_shape_vec((shape[0], shape[1]), data.clone())
            .with_context(|| format!("reshaping {name}"))
    }
}
