//! Hand-rolled forward for the transformer student — a 1:1 port of
//! `StudentModel.forward_dense` (batch = 1, no padding, dropout off).
//!
//! Charge is factored out of the trunk (RT stays charge-invariant) and re-enters at the heads:
//! added per fragment site for MS2, concatenated for CCS. `ms_context` shifts the MS2 fragment
//! features; `chrom_context` shifts the RT head; CCS takes no context.

use anyhow::{anyhow, Result};
use ndarray::{s, Array1, Array2};

use crate::artifact::Artifact;

fn gelu(x: f32) -> f32 {
    let xf = x as f64;
    (0.5 * xf * (1.0 + libm::erf(xf / std::f64::consts::SQRT_2))) as f32
}

fn act_scalar(v: f32, act: &str) -> f32 {
    match act {
        "relu" => v.max(0.0),
        _ => gelu(v),
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// y = x @ Wᵀ + b, with x [n,in], W [out,in], b [out].
fn linear(x: &Array2<f32>, w: &Array2<f32>, b: &Array1<f32>) -> Array2<f32> {
    let mut y = x.dot(&w.t());
    for mut row in y.rows_mut() {
        for (r, &bb) in row.iter_mut().zip(b.iter()) {
            *r += bb;
        }
    }
    y
}

/// y = W @ x + b, with x [in], W [out,in], b [out].
fn linear1(x: &Array1<f32>, w: &Array2<f32>, b: &Array1<f32>) -> Array1<f32> {
    let mut y = w.dot(x);
    for (r, &bb) in y.iter_mut().zip(b.iter()) {
        *r += bb;
    }
    y
}

/// LayerNorm over the last dim (biased variance, eps inside sqrt) — matches torch nn.LayerNorm.
fn layernorm(x: &Array2<f32>, g: &Array1<f32>, b: &Array1<f32>) -> Array2<f32> {
    const EPS: f32 = 1e-5;
    let mut y = x.clone();
    let n = x.ncols() as f32;
    for mut row in y.rows_mut() {
        let mean = row.sum() / n;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let denom = (var + EPS).sqrt();
        for (i, r) in row.iter_mut().enumerate() {
            *r = g[i] * ((*r - mean) / denom) + b[i];
        }
    }
    y
}

fn softmax_rows(mut a: Array2<f32>) -> Array2<f32> {
    for mut row in a.rows_mut() {
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            s += *v;
        }
        for v in row.iter_mut() {
            *v /= s;
        }
    }
    a
}

pub struct Predictor<'a> {
    art: &'a Artifact,
}

impl<'a> Predictor<'a> {
    pub fn new(art: &'a Artifact) -> Self {
        Self { art }
    }

    fn cfg(&self) -> &crate::artifact::Config {
        &self.art.meta.config
    }

    /// Multi-head self-attention for one layer (dense, no mask).
    fn mha(&self, x: &Array2<f32>, l: usize) -> Result<Array2<f32>> {
        let d = self.cfg().d_model;
        let h = self.cfg().n_heads;
        let dk = d / h;
        let p = format!("model.backbone.net.layers.{l}");
        let in_w = self.art.get2(&format!("{p}.self_attn.in_proj_weight"))?; // [3d,d]
        let in_b = self.art.get1(&format!("{p}.self_attn.in_proj_bias"))?; // [3d]
        let out_w = self.art.get2(&format!("{p}.self_attn.out_proj.weight"))?; // [d,d]
        let out_b = self.art.get1(&format!("{p}.self_attn.out_proj.bias"))?; // [d]

        let qkv = linear(x, &in_w, &in_b); // [t,3d]
        let q = qkv.slice(s![.., 0..d]).to_owned();
        let k = qkv.slice(s![.., d..2 * d]).to_owned();
        let v = qkv.slice(s![.., 2 * d..3 * d]).to_owned();

        let t = x.nrows();
        let scale = 1.0 / (dk as f32).sqrt();
        let mut ctx = Array2::<f32>::zeros((t, d));
        for head in 0..h {
            let c0 = head * dk;
            let c1 = c0 + dk;
            let qh = q.slice(s![.., c0..c1]);
            let kh = k.slice(s![.., c0..c1]);
            let vh = v.slice(s![.., c0..c1]);
            let scores = softmax_rows(qh.dot(&kh.t()) * scale); // [t,t]
            let ctx_h = scores.dot(&vh); // [t,dk]
            ctx.slice_mut(s![.., c0..c1]).assign(&ctx_h);
        }
        Ok(linear(&ctx, &out_w, &out_b))
    }

    fn layer(&self, x: Array2<f32>, l: usize) -> Result<Array2<f32>> {
        let p = format!("model.backbone.net.layers.{l}");
        let act = self.cfg().activation.clone();
        // Post-LN block: x = LN(x + MHA(x)); x = LN(x + FFN(x)).
        let attn = self.mha(&x, l)?;
        let x = layernorm(
            &(&x + &attn),
            &self.art.get1(&format!("{p}.norm1.weight"))?,
            &self.art.get1(&format!("{p}.norm1.bias"))?,
        );
        let l1 = linear(
            &x,
            &self.art.get2(&format!("{p}.linear1.weight"))?,
            &self.art.get1(&format!("{p}.linear1.bias"))?,
        );
        let mut l1a = l1;
        l1a.mapv_inplace(|v| act_scalar(v, &act));
        let ff = linear(
            &l1a,
            &self.art.get2(&format!("{p}.linear2.weight"))?,
            &self.art.get1(&format!("{p}.linear2.bias"))?,
        );
        Ok(layernorm(
            &(&x + &ff),
            &self.art.get1(&format!("{p}.norm2.weight"))?,
            &self.art.get1(&format!("{p}.norm2.bias"))?,
        ))
    }

    fn head(&self, x: &Array2<f32>, prefix: &str) -> Result<Array2<f32>> {
        let act = self.cfg().activation.clone();
        let mut h0 = linear(
            x,
            &self.art.get2(&format!("{prefix}.0.weight"))?,
            &self.art.get1(&format!("{prefix}.0.bias"))?,
        );
        h0.mapv_inplace(|v| act_scalar(v, &act));
        Ok(linear(
            &h0,
            &self.art.get2(&format!("{prefix}.2.weight"))?,
            &self.art.get1(&format!("{prefix}.2.bias"))?,
        ))
    }

    /// Build ms_context from acquisition factors (MSContextEncoder forward). Unknown categorical
    /// names resolve to id 0 (the neutral row). `energy = None` omits the energy term.
    pub fn encode_ms_context(
        &self,
        instrument: &str,
        detector: &str,
        fragmentation: &str,
        energy: Option<f32>,
    ) -> Result<Array1<f32>> {
        let vocab = self
            .art
            .meta
            .vocab
            .as_ref()
            .ok_or_else(|| anyhow!("artifact has no acquisition encoder"))?;
        let idx = |list: &[String], name: &str| list.iter().position(|n| n == name).unwrap_or(0);
        let inst = self.art.get2("enc.inst_emb.weight")?;
        let det = self.art.get2("enc.det_emb.weight")?;
        let frag = self.art.get2("enc.frag_emb.weight")?;
        let mut ctx = &inst.row(idx(&vocab.instruments, instrument)).to_owned()
            + &det.row(idx(&vocab.detectors, detector)).to_owned();
        ctx = &ctx
            + &frag
                .row(idx(&vocab.fragmentations, fragmentation))
                .to_owned();
        if let Some(e) = energy {
            // energy_mlp: Linear(1,d) -> GELU -> Linear(d,d); always GELU regardless of cfg.
            let w0 = self.art.get2("enc.energy_mlp.0.weight")?; // [d,1]
            let b0 = self.art.get1("enc.energy_mlp.0.bias")?;
            let w2 = self.art.get2("enc.energy_mlp.2.weight")?; // [d,d]
            let b2 = self.art.get1("enc.energy_mlp.2.bias")?;
            let mut hidden = &w0.column(0).to_owned() * e + &b0;
            hidden.mapv_inplace(gelu);
            ctx = &ctx + &linear1(&hidden, &w2, &b2);
        }
        Ok(ctx)
    }

    /// chrom_context for a named dataset (ChromRunbook row lookup).
    pub fn chrom_context(&self, dataset: &str) -> Result<Array1<f32>> {
        let index = self
            .art
            .meta
            .dataset_index
            .as_ref()
            .ok_or_else(|| anyhow!("artifact has no dataset index"))?;
        let row = *index
            .get(dataset)
            .ok_or_else(|| anyhow!("unknown --chrom-context {dataset:?}; known: {index:?}"))?;
        Ok(self
            .art
            .get2("runbook.emb.weight")?
            .row(row as usize)
            .to_owned())
    }

    /// Forward pass. Returns (ms2 [L-1, n_ion] in (0,1), rt native, ccs native).
    pub fn forward(
        &self,
        seq: &[u8],
        charge: i64,
        ms_ctx: Option<&Array1<f32>>,
        chrom_ctx: Option<&Array1<f32>>,
    ) -> Result<(Array2<f32>, f32, f32)> {
        let cfg = self.cfg();
        let d = cfg.d_model;
        let t = seq.len();

        // --- embed: token + position + mod_proj(0) (the mod bias always applies) ---
        let token_emb = self.art.get2("model.token_emb.weight")?;
        let pos_emb = self.art.get2("model.pos_emb.weight")?;
        let mod_w = self.art.get2("model.mod_proj.weight")?; // [d,1]
        let mod_b = self.art.get1("model.mod_proj.bias")?;
        let mut x = Array2::<f32>::zeros((t, d));
        for i in 0..t {
            let tok = (seq[i] - b'A') as usize; // token id = ord(aa) - ord('A')
            for j in 0..d {
                // mod_delta = 0 (v1 unmodified) -> mod_proj(0) = mod_b[j].
                x[[i, j]] = token_emb[[tok, j]] + pos_emb[[i, j]] + mod_w[[j, 0]] * 0.0 + mod_b[j];
            }
        }

        // --- transformer backbone ---
        for l in 0..cfg.n_layers {
            x = self.layer(x, l)?;
        }

        // --- pooled peptide rep + charge embedding ---
        let pooled = x.mean_axis(ndarray::Axis(0)).unwrap(); // [d]
        let ce = self
            .art
            .get2("model.charge_emb.weight")?
            .row(charge as usize)
            .to_owned();

        // --- MS2 head: adjacent-pool fragment features + ms_context + charge (added) ---
        let mut frag = Array2::<f32>::zeros((t - 1, d));
        for i in 0..t - 1 {
            for j in 0..d {
                frag[[i, j]] = 0.5 * (x[[i, j]] + x[[i + 1, j]]);
            }
        }
        if let Some(ctx) = ms_ctx {
            let shift = linear1(
                ctx,
                &self.art.get2("model.ms_to_frag.weight")?,
                &self.art.get1("model.ms_to_frag.bias")?,
            );
            for mut row in frag.rows_mut() {
                for (r, &sh) in row.iter_mut().zip(shift.iter()) {
                    *r += sh;
                }
            }
        }
        for mut row in frag.rows_mut() {
            for (r, &cc) in row.iter_mut().zip(ce.iter()) {
                *r += cc;
            }
        }
        let mut ms2 = self.head(&frag, "model.ms2_head")?; // [t-1, n_ion]
        ms2.mapv_inplace(sigmoid);

        // --- CCS head: concat[pooled, charge] ---
        let mut ccs_in = Array1::<f32>::zeros(2 * d);
        ccs_in.slice_mut(s![0..d]).assign(&pooled);
        ccs_in.slice_mut(s![d..2 * d]).assign(&ce);
        let ccs_2d = ccs_in.insert_axis(ndarray::Axis(0)); // [1,2d]
        let ccs_out = self.head(&ccs_2d, "model.ccs_head")?;
        let ccs = ccs_out[[0, 0]] * self.art.meta.norm.ccs_std + self.art.meta.norm.ccs_mean;

        // --- RT head: pooled (+ chrom_context); never sees charge ---
        let mut rt_feat = pooled.clone();
        if let Some(ctx) = chrom_ctx {
            let shift = linear1(
                ctx,
                &self.art.get2("model.chrom_to_rt.weight")?,
                &self.art.get1("model.chrom_to_rt.bias")?,
            );
            rt_feat = &rt_feat + &shift;
        }
        let rt_2d = rt_feat.insert_axis(ndarray::Axis(0)); // [1,d]
        let rt_out = self.head(&rt_2d, "model.rt_head")?;
        let rt = rt_out[[0, 0]] * self.art.meta.norm.rt_std + self.art.meta.norm.rt_mean;

        Ok((ms2, rt, ccs))
    }
}
