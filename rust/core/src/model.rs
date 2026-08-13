//! Hand-rolled forward for the transformer student — a 1:1 port of
//! `StudentModel.forward_dense` (same-length dense batches, no padding, dropout off).
//!
//! Charge is factored out of the trunk (RT stays charge-invariant) and re-enters at the heads:
//! added per fragment site for MS2, concatenated for CCS. `ms_context` shifts the MS2 fragment
//! features; `chrom_context` shifts the RT head; CCS takes no context.

use anyhow::{anyhow, Result};
use ndarray::{s, Array1, Array2, Array3, ArrayBase, Data, Ix1, Ix2};

use crate::artifact::Artifact;
use crate::peptide::Peptide;
use crate::tokenize::{self, CTERM_IDX, FRAG_OFFSET, NTERM_IDX};

fn gelu(x: f32) -> f32 {
    let xf = x as f64;
    (0.5 * xf * (1.0 + libm::erf(xf / std::f64::consts::SQRT_2))) as f32
}

/// The tanh approximation used by PyTorch's `GELU(approximate="tanh")`.
fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
}

fn act_scalar(v: f32, act: &str) -> f32 {
    match act {
        "relu" => v.max(0.0),
        "leaky_relu" => {
            if v < 0.0 {
                0.01 * v
            } else {
                v
            }
        }
        "gelu_tanh" => gelu_tanh(v),
        _ => gelu(v),
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// y = x @ Wᵀ + b, with x [n,in], W [out,in], b [out].
fn linear<SX, SW, SB>(
    x: &ArrayBase<SX, Ix2>,
    w: &ArrayBase<SW, Ix2>,
    b: &ArrayBase<SB, Ix1>,
) -> Array2<f32>
where
    SX: Data<Elem = f32>,
    SW: Data<Elem = f32>,
    SB: Data<Elem = f32>,
{
    let mut y = x.dot(&w.t());
    for mut row in y.rows_mut() {
        for (r, &bb) in row.iter_mut().zip(b.iter()) {
            *r += bb;
        }
    }
    y
}

/// y = W @ x + b, with x [in], W [out,in], b [out].
fn linear1<SX, SW, SB>(
    x: &ArrayBase<SX, Ix1>,
    w: &ArrayBase<SW, Ix2>,
    b: &ArrayBase<SB, Ix1>,
) -> Array1<f32>
where
    SX: Data<Elem = f32>,
    SW: Data<Elem = f32>,
    SB: Data<Elem = f32>,
{
    let mut y = w.dot(x);
    for (r, &bb) in y.iter_mut().zip(b.iter()) {
        *r += bb;
    }
    y
}

/// LayerNorm over the last dim (biased variance, eps inside sqrt) — matches torch nn.LayerNorm.
fn layernorm<SX, SG, SB>(
    x: &ArrayBase<SX, Ix2>,
    g: &ArrayBase<SG, Ix1>,
    b: &ArrayBase<SB, Ix1>,
) -> Array2<f32>
where
    SX: Data<Elem = f32>,
    SG: Data<Elem = f32>,
    SB: Data<Elem = f32>,
{
    const EPS: f32 = 1e-5;
    let mut y = x.to_owned();
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

/// Charge-independent transformer output for one modified peptide.
pub struct EncodedPeptide {
    pooled: Array1<f32>,
    frag: Array2<f32>,
}

/// Charge-independent transformer output for a same-length peptide batch.
pub struct EncodedPeptideBatch {
    pooled: Array2<f32>,
    frag: Array3<f32>,
}

pub type ChargeOutputs = Vec<(Array2<f32>, f32)>;

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

    /// Batched projection/FFN path for same-length peptides. Attention remains isolated per
    /// peptide, but all dense projections operate on `(batch * tokens)` rows in one GEMM.
    fn mha_batch(&self, x: &Array3<f32>, l: usize) -> Result<Array3<f32>> {
        let (batch, t, d) = x.dim();
        let h = self.cfg().n_heads;
        let dk = d / h;
        let p = format!("model.backbone.net.layers.{l}");
        let flat = x
            .view()
            .into_shape_with_order((batch * t, d))
            .map_err(|e| anyhow!("flattening attention batch: {e}"))?;
        let qkv = linear(
            &flat,
            &self.art.get2(&format!("{p}.self_attn.in_proj_weight"))?,
            &self.art.get1(&format!("{p}.self_attn.in_proj_bias"))?,
        )
        .into_shape_with_order((batch, t, 3 * d))
        .map_err(|e| anyhow!("reshaping batched qkv: {e}"))?;

        let scale = 1.0 / (dk as f32).sqrt();
        let mut ctx = Array3::<f32>::zeros((batch, t, d));
        for batch_i in 0..batch {
            for head in 0..h {
                let c0 = head * dk;
                let c1 = c0 + dk;
                let qh = qkv.slice(s![batch_i, .., c0..c1]);
                let kh = qkv.slice(s![batch_i, .., d + c0..d + c1]);
                let vh = qkv.slice(s![batch_i, .., 2 * d + c0..2 * d + c1]);
                let scores = softmax_rows(qh.dot(&kh.t()) * scale);
                let ctx_h = scores.dot(&vh);
                ctx.slice_mut(s![batch_i, .., c0..c1]).assign(&ctx_h);
            }
        }
        let ctx_flat = ctx
            .view()
            .into_shape_with_order((batch * t, d))
            .map_err(|e| anyhow!("flattening batched attention output: {e}"))?;
        linear(
            &ctx_flat,
            &self.art.get2(&format!("{p}.self_attn.out_proj.weight"))?,
            &self.art.get1(&format!("{p}.self_attn.out_proj.bias"))?,
        )
        .into_shape_with_order((batch, t, d))
        .map_err(|e| anyhow!("reshaping batched attention output: {e}"))
    }

    fn layer_batch(&self, x: Array3<f32>, l: usize) -> Result<Array3<f32>> {
        let (batch, t, d) = x.dim();
        let p = format!("model.backbone.net.layers.{l}");
        let act = self.cfg().activation.clone();
        let attn = self.mha_batch(&x, l)?;
        let residual = &x + &attn;
        let residual_flat = residual
            .view()
            .into_shape_with_order((batch * t, d))
            .map_err(|e| anyhow!("flattening norm1 batch: {e}"))?;
        let x = layernorm(
            &residual_flat,
            &self.art.get1(&format!("{p}.norm1.weight"))?,
            &self.art.get1(&format!("{p}.norm1.bias"))?,
        )
        .into_shape_with_order((batch, t, d))
        .map_err(|e| anyhow!("reshaping norm1 batch: {e}"))?;
        let x_flat = x
            .view()
            .into_shape_with_order((batch * t, d))
            .map_err(|e| anyhow!("flattening FFN batch: {e}"))?;
        let mut l1 = linear(
            &x_flat,
            &self.art.get2(&format!("{p}.linear1.weight"))?,
            &self.art.get1(&format!("{p}.linear1.bias"))?,
        );
        l1.mapv_inplace(|v| act_scalar(v, &act));
        let ff = linear(
            &l1,
            &self.art.get2(&format!("{p}.linear2.weight"))?,
            &self.art.get1(&format!("{p}.linear2.bias"))?,
        )
        .into_shape_with_order((batch, t, d))
        .map_err(|e| anyhow!("reshaping FFN batch: {e}"))?;
        let residual = &x + &ff;
        let residual_flat = residual
            .view()
            .into_shape_with_order((batch * t, d))
            .map_err(|e| anyhow!("flattening norm2 batch: {e}"))?;
        layernorm(
            &residual_flat,
            &self.art.get1(&format!("{p}.norm2.weight"))?,
            &self.art.get1(&format!("{p}.norm2.bias"))?,
        )
        .into_shape_with_order((batch, t, d))
        .map_err(|e| anyhow!("reshaping norm2 batch: {e}"))
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
        let mut ctx = Array1::zeros(inst.ncols());
        for (table, row) in [
            (&inst, idx(&vocab.instruments, instrument)),
            (&det, idx(&vocab.detectors, detector)),
            (&frag, idx(&vocab.fragmentations, fragmentation)),
        ] {
            if row != 0 {
                ctx = &ctx + &table.row(row);
            }
        }
        if let Some(e) = energy {
            // energy_mlp: Linear(1,d) -> GELU -> Linear(d,d); always GELU regardless of cfg.
            let w0 = self.art.get2("enc.energy_mlp.0.weight")?; // [d,1]
            let b0 = self.art.get1("enc.energy_mlp.0.bias")?;
            let w2 = self.art.get2("enc.energy_mlp.2.weight")?; // [d,d]
            let b2 = self.art.get1("enc.energy_mlp.2.bias")?;
            let mut hidden = &w0.column(0).to_owned() * e + b0;
            hidden.mapv_inplace(gelu);
            ctx = &ctx + &linear1(&hidden, &w2, &b2);
        }
        Ok(ctx)
    }

    /// Resolve acquisition context all the way to the additive fragment-feature shift.
    /// This projection is peptide- and charge-independent and should be cached by bulk callers.
    pub fn ms_context_shift(
        &self,
        instrument: &str,
        detector: &str,
        fragmentation: &str,
        energy: Option<f32>,
    ) -> Result<Array1<f32>> {
        let ctx = self.encode_ms_context(instrument, detector, fragmentation, energy)?;
        Ok(linear1(
            &ctx,
            &self.art.get2("model.ms_to_frag.weight")?,
            &self.art.get1("model.ms_to_frag.bias")?,
        ))
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

    /// Resolve chromatography context to the additive RT-feature shift. Like MS context, this
    /// depends only on the selected dataset and can be prepared once for a whole library.
    pub fn chrom_context_shift(&self, dataset: &str) -> Result<Array1<f32>> {
        let ctx = self.chrom_context(dataset)?;
        Ok(linear1(
            &ctx,
            &self.art.get2("model.chrom_to_rt.weight")?,
            &self.art.get1("model.chrom_to_rt.bias")?,
        ))
    }

    /// Per-dataset RT output affine `(scale, shift)` — the ChromRunbook's scale+shift row.
    ///
    /// `chrom_context` above is an additive bias in feature space and cannot express a
    /// rescale; this carries the part of a dataset's raw-RT relationship that is a global
    /// scale (gradient length, unit differences). Applied to the RT head's output in
    /// standardized space, before denormalization, exactly as torch does it.
    pub fn chrom_affine(&self, dataset: &str) -> Result<(f32, f32)> {
        let index = self
            .art
            .meta
            .dataset_index
            .as_ref()
            .ok_or_else(|| anyhow!("artifact has no dataset index"))?;
        let row = *index
            .get(dataset)
            .ok_or_else(|| anyhow!("unknown --chrom-context {dataset:?}; known: {index:?}"))?
            as usize;
        let log_scale = self.art.get2("runbook.log_scale.weight")?[[row, 0]];
        let shift = self.art.get2("runbook.shift.weight")?[[row, 0]];
        Ok((log_scale.exp(), shift))
    }

    /// `comp_enc`: `Linear(N_ELEMENTS, d)` over the element-composition delta (C,H,N,O,S,P).
    fn comp_vec(&self, comp: ndarray::ArrayView1<f32>) -> Result<Array1<f32>> {
        let w = self.art.get2("model.comp_enc.weight")?; // [d, N_ELEMENTS]
        let b = self.art.get1("model.comp_enc.bias")?; // [d]
        if w.ncols() != comp.len() {
            return Err(anyhow!(
                "comp_enc expects {} elements, got {}",
                w.ncols(),
                comp.len()
            ));
        }
        Ok(linear1(&comp.to_owned(), &w, &b))
    }

    /// `mass_enc`: FourierFeatures -> Linear -> activation -> Linear, over an unscaled Dalton
    /// delta.
    ///
    /// The frequency ladder is read from the artifact buffer (`mass_enc.0.freq`) rather than
    /// recomputed from WAVELENGTH_MIN/MAX here: a reconstructed ladder would be free to drift
    /// away from the torch one, and the two runtimes would disagree silently.
    fn mass_vec(&self, mass: f32) -> Result<Array1<f32>> {
        let freq = self.art.get1("model.mass_enc.0.freq")?; // [k]
        let k = freq.len();
        let mut feat = Array1::<f32>::zeros(2 * k);
        for i in 0..k {
            let a = mass * freq[i];
            feat[i] = a.sin();
            feat[k + i] = a.cos();
        }
        let act = self.cfg().activation.clone();
        let mut h = linear1(
            &feat,
            &self.art.get2("model.mass_enc.1.weight")?, // [d, 2k]
            &self.art.get1("model.mass_enc.1.bias")?,
        );
        h.mapv_inplace(|v| act_scalar(v, &act));
        Ok(linear1(
            &h,
            &self.art.get2("model.mass_enc.3.weight")?, // [d, d]
            &self.art.get1("model.mass_enc.3.bias")?,
        ))
    }

    /// Encode one modified peptide through the shared transformer trunk.
    pub fn encode(&self, pep: &Peptide) -> Result<EncodedPeptide> {
        let cfg = self.cfg();
        let d = cfg.d_model;
        let seq = pep.sequence.as_bytes();
        // Guarded here, not just in `predict`: `encode` is public, and below `seq.len() - 1`
        // would wrap to usize::MAX on an empty peptide and abort in an allocation.
        if seq.len() < 2 {
            return Err(anyhow!(
                "peptide {:?} has {} residue(s); MS2 needs at least 2",
                pep.sequence,
                seq.len()
            ));
        }
        // Token layout is always [N] r1..rL [C], so T = L + 2 and residue i sits at column 1+i.
        let t = seq.len() + 2;
        if t > cfg.max_len {
            return Err(anyhow!(
                "peptide of length {} needs {t} token columns, but the model's pos_emb holds \
                 only max_len={}",
                seq.len(),
                cfg.max_len
            ));
        }
        // --- embed: token + position + routed mod vector ---
        // Eval routing sends named mods through comp_enc and mass-only mods through mass_enc,
        // never both; unmodified columns (termini included) contribute exactly zero (there is
        // no unconditional bias term — the encoder only runs where mod_present).
        //
        // The four channels come from `tokenize::mod_arrays`, the same builder the torch path
        // uses through the pyo3 ext, so this ACCUMULATES co-sited mods into one channel value
        // and encodes once. Encoding each mod separately and summing the vectors would add
        // comp_enc.bias once per mod, which torch does not do.
        let mods = tokenize::mod_arrays(std::slice::from_ref(pep), t)?;
        let token_emb = self.art.get2("model.token_emb.weight")?;
        let pos_emb = self.art.get2("model.pos_emb.weight")?;
        let mut x = Array2::<f32>::zeros((t, d));
        for i in 0..t {
            let tok = match i {
                0 => NTERM_IDX as usize,
                _ if i == t - 1 => CTERM_IDX as usize,
                _ => (seq[i - 1] - b'A') as usize, // token id = ord(aa) - ord('A')
            };
            for j in 0..d {
                x[[i, j]] = token_emb[[tok, j]] + pos_emb[[i, j]];
            }
        }
        for i in 0..t {
            if !mods.mod_present[[0, i]] {
                continue;
            }
            let v = if mods.mod_has_composition[[0, i]] {
                self.comp_vec(mods.mod_comp.slice(s![0, i, ..]))?
            } else {
                self.mass_vec(mods.mod_mass[[0, i]])?
            };
            for j in 0..d {
                x[[i, j]] += v[j];
            }
        }

        // --- transformer backbone ---
        for l in 0..cfg.n_layers {
            x = self.layer(x, l)?;
        }

        // --- pooled peptide and adjacent-pool fragment representations ---
        let pooled = x.mean_axis(ndarray::Axis(0)).unwrap(); // [d]
                                                             // Adjacent-pool row p covers tokens (p, p+1). With the mandatory N-term token at column
                                                             // 0, the first inter-RESIDUE site is row 1 (residues 1 and 2), so the L-1 real fragment
                                                             // sites are rows [1, L). Rows 0 and L (the N-/C-term pools) are dropped, exactly as
                                                             // `predict_library_fast` slices them off with the same `FRAG_OFFSET`, re-exported to
                                                             // Python by the pyo3 ext so the two runtimes cannot drift.
        let frag_pos = seq.len() - 1;
        let mut frag = Array2::<f32>::zeros((frag_pos, d));
        for i in 0..frag_pos {
            let p = FRAG_OFFSET + i;
            for j in 0..d {
                frag[[i, j]] = 0.5 * (x[[p, j]] + x[[p + 1, j]]);
            }
        }
        Ok(EncodedPeptide { pooled, frag })
    }

    /// Encode a non-empty batch whose peptides all have the same residue length.
    pub fn encode_batch(&self, peptides: &[Peptide]) -> Result<EncodedPeptideBatch> {
        let first = peptides
            .first()
            .ok_or_else(|| anyhow!("cannot encode an empty peptide batch"))?;
        let seq_len = first.sequence.len();
        if seq_len < 2 {
            return Err(anyhow!(
                "peptide batch contains a sequence shorter than 2 residues"
            ));
        }
        if peptides.iter().any(|pep| pep.sequence.len() != seq_len) {
            return Err(anyhow!("encode_batch requires one shared peptide length"));
        }
        let cfg = self.cfg();
        let batch = peptides.len();
        let t = seq_len + 2;
        let d = cfg.d_model;
        if t > cfg.max_len {
            return Err(anyhow!(
                "peptide length {seq_len} needs {t} token columns, but max_len={}",
                cfg.max_len
            ));
        }

        let mods = tokenize::mod_arrays(peptides, t)?;
        let token_emb = self.art.get2("model.token_emb.weight")?;
        let pos_emb = self.art.get2("model.pos_emb.weight")?;
        let mut x = Array3::<f32>::zeros((batch, t, d));
        for (batch_i, pep) in peptides.iter().enumerate() {
            let seq = pep.sequence.as_bytes();
            for i in 0..t {
                let tok = match i {
                    0 => NTERM_IDX as usize,
                    _ if i == t - 1 => CTERM_IDX as usize,
                    _ => (seq[i - 1] - b'A') as usize,
                };
                for j in 0..d {
                    x[[batch_i, i, j]] = token_emb[[tok, j]] + pos_emb[[i, j]];
                }
            }
            for i in 0..t {
                if !mods.mod_present[[batch_i, i]] {
                    continue;
                }
                let v = if mods.mod_has_composition[[batch_i, i]] {
                    self.comp_vec(mods.mod_comp.slice(s![batch_i, i, ..]))?
                } else {
                    self.mass_vec(mods.mod_mass[[batch_i, i]])?
                };
                for j in 0..d {
                    x[[batch_i, i, j]] += v[j];
                }
            }
        }
        for l in 0..cfg.n_layers {
            x = self.layer_batch(x, l)?;
        }

        let mut pooled = Array2::<f32>::zeros((batch, d));
        let frag_pos = seq_len - 1;
        let mut frag = Array3::<f32>::zeros((batch, frag_pos, d));
        for batch_i in 0..batch {
            pooled.row_mut(batch_i).assign(
                &x.slice(s![batch_i, .., ..])
                    .mean_axis(ndarray::Axis(0))
                    .unwrap(),
            );
            for i in 0..frag_pos {
                let p = FRAG_OFFSET + i;
                for j in 0..d {
                    frag[[batch_i, i, j]] = 0.5 * (x[[batch_i, p, j]] + x[[batch_i, p + 1, j]]);
                }
            }
        }
        Ok(EncodedPeptideBatch { pooled, frag })
    }

    /// Run the charge-dependent MS2 and CCS heads over an encoded peptide.
    pub fn predict_charge(
        &self,
        encoded: &EncodedPeptide,
        charge: i64,
        ms_shift: Option<&Array1<f32>>,
    ) -> Result<(Array2<f32>, f32)> {
        let mut outputs = self.predict_charges(encoded, &[charge], ms_shift)?;
        Ok(outputs
            .pop()
            .expect("one requested charge yields one output"))
    }

    /// Run all requested charge heads in two larger matrix multiplications. Rows remain grouped
    /// by charge, so callers can split the MS2 output without changing transition ordering.
    pub fn predict_charges(
        &self,
        encoded: &EncodedPeptide,
        charges: &[i64],
        ms_shift: Option<&Array1<f32>>,
    ) -> Result<Vec<(Array2<f32>, f32)>> {
        if charges.is_empty() {
            return Ok(Vec::new());
        }
        let cfg = self.cfg();
        let d = cfg.d_model;
        let frag_pos = encoded.frag.nrows();
        let charge_emb = self.art.get2("model.charge_emb.weight")?;
        let mut ce = Array2::<f32>::zeros((charges.len(), d));
        for (i, &charge) in charges.iter().enumerate() {
            // charge_emb has max_charge + 1 rows. Validate before converting to usize so a
            // negative charge cannot wrap into an ndarray bounds assertion.
            if charge < 1 || charge as usize > cfg.max_charge {
                return Err(anyhow!(
                    "charge {charge} is out of range; this model's charge_emb covers 1..={}",
                    cfg.max_charge
                ));
            }
            ce.row_mut(i).assign(&charge_emb.row(charge as usize));
        }

        let mut frag = Array2::<f32>::zeros((charges.len() * frag_pos, d));
        for charge_i in 0..charges.len() {
            for frag_i in 0..frag_pos {
                let mut row = frag.row_mut(charge_i * frag_pos + frag_i);
                row.assign(&encoded.frag.row(frag_i));
                if let Some(shift) = ms_shift {
                    row += shift;
                }
                row += &ce.row(charge_i);
            }
        }
        let mut ms2 = self.head(&frag, "model.ms2_head")?; // [charge * (L-1), n_ion]
        ms2.mapv_inplace(sigmoid);

        // --- CCS head: concat[pooled, charge] ---
        let mut ccs_in = Array2::<f32>::zeros((charges.len(), 2 * d));
        for charge_i in 0..charges.len() {
            ccs_in.slice_mut(s![charge_i, 0..d]).assign(&encoded.pooled);
            ccs_in
                .slice_mut(s![charge_i, d..2 * d])
                .assign(&ce.row(charge_i));
        }
        let ccs_out = self.head(&ccs_in, "model.ccs_head")?;

        let mut outputs = Vec::with_capacity(charges.len());
        for charge_i in 0..charges.len() {
            let spectrum = ms2
                .slice(s![charge_i * frag_pos..(charge_i + 1) * frag_pos, ..])
                .to_owned();
            let ccs =
                ccs_out[[charge_i, 0]] * self.art.meta.norm.ccs_std + self.art.meta.norm.ccs_mean;
            outputs.push((spectrum, ccs));
        }
        Ok(outputs)
    }

    /// Run charge-specific heads over every peptide in an encoded batch. Output nesting is
    /// `[peptide][charge]`, matching the input orders.
    pub fn predict_batch_charges(
        &self,
        encoded: &EncodedPeptideBatch,
        charges: &[i64],
        ms_shift: Option<&Array1<f32>>,
    ) -> Result<Vec<ChargeOutputs>> {
        let (batch, frag_pos, d) = encoded.frag.dim();
        if charges.is_empty() {
            return Ok((0..batch).map(|_| Vec::new()).collect());
        }
        let cfg = self.cfg();
        let charge_emb = self.art.get2("model.charge_emb.weight")?;
        let mut ce = Array2::<f32>::zeros((charges.len(), d));
        for (charge_i, &charge) in charges.iter().enumerate() {
            if charge < 1 || charge as usize > cfg.max_charge {
                return Err(anyhow!(
                    "charge {charge} is out of range; this model's charge_emb covers 1..={}",
                    cfg.max_charge
                ));
            }
            ce.row_mut(charge_i)
                .assign(&charge_emb.row(charge as usize));
        }

        let mut frag = Array2::<f32>::zeros((batch * charges.len() * frag_pos, d));
        let mut ccs_in = Array2::<f32>::zeros((batch * charges.len(), 2 * d));
        for batch_i in 0..batch {
            for charge_i in 0..charges.len() {
                let pair_i = batch_i * charges.len() + charge_i;
                ccs_in
                    .slice_mut(s![pair_i, 0..d])
                    .assign(&encoded.pooled.row(batch_i));
                ccs_in
                    .slice_mut(s![pair_i, d..2 * d])
                    .assign(&ce.row(charge_i));
                for frag_i in 0..frag_pos {
                    let mut row = frag.row_mut(pair_i * frag_pos + frag_i);
                    row.assign(&encoded.frag.slice(s![batch_i, frag_i, ..]));
                    if let Some(shift) = ms_shift {
                        row += shift;
                    }
                    row += &ce.row(charge_i);
                }
            }
        }
        let mut ms2 = self.head(&frag, "model.ms2_head")?;
        ms2.mapv_inplace(sigmoid);
        let ccs_out = self.head(&ccs_in, "model.ccs_head")?;

        let mut outputs = Vec::with_capacity(batch);
        for batch_i in 0..batch {
            let mut peptide_outputs = Vec::with_capacity(charges.len());
            for charge_i in 0..charges.len() {
                let pair_i = batch_i * charges.len() + charge_i;
                let spectrum = ms2
                    .slice(s![pair_i * frag_pos..(pair_i + 1) * frag_pos, ..])
                    .to_owned();
                let ccs =
                    ccs_out[[pair_i, 0]] * self.art.meta.norm.ccs_std + self.art.meta.norm.ccs_mean;
                peptide_outputs.push((spectrum, ccs));
            }
            outputs.push(peptide_outputs);
        }
        Ok(outputs)
    }

    /// Run the charge-independent RT head over an encoded peptide.
    pub fn predict_rt(
        &self,
        encoded: &EncodedPeptide,
        chrom_shift: Option<&Array1<f32>>,
        chrom_affine: Option<(f32, f32)>,
    ) -> Result<f32> {
        let mut rt_feat = encoded.pooled.clone();
        if let Some(shift) = chrom_shift {
            rt_feat = &rt_feat + shift;
        }
        let rt_2d = rt_feat.insert_axis(ndarray::Axis(0)); // [1,d]
        let rt_out = self.head(&rt_2d, "model.rt_head")?;
        // Per-dataset affine applies in STANDARDIZED space, before denormalization — same
        // order as torch, where `rt = scale * rt_head(feat) + shift` is compared against
        // standardize_rt(raw_rt). Applying it after denormalization would rescale the frame
        // itself and silently disagree with the training-time semantics.
        let rt_std_space = match chrom_affine {
            Some((scale, shift)) => scale * rt_out[[0, 0]] + shift,
            None => rt_out[[0, 0]],
        };
        let rt = rt_std_space * self.art.meta.norm.rt_std + self.art.meta.norm.rt_mean;

        Ok(rt)
    }

    pub fn predict_rt_batch(
        &self,
        encoded: &EncodedPeptideBatch,
        chrom_shift: Option<&Array1<f32>>,
        chrom_affine: Option<(f32, f32)>,
    ) -> Result<Array1<f32>> {
        let mut rt_feat = encoded.pooled.clone();
        if let Some(shift) = chrom_shift {
            for mut row in rt_feat.rows_mut() {
                row += shift;
            }
        }
        let rt_out = self.head(&rt_feat, "model.rt_head")?;
        let mut rt = Array1::<f32>::zeros(rt_out.nrows());
        for i in 0..rt.len() {
            let standardized = match chrom_affine {
                Some((scale, shift)) => scale * rt_out[[i, 0]] + shift,
                None => rt_out[[i, 0]],
            };
            rt[i] = standardized * self.art.meta.norm.rt_std + self.art.meta.norm.rt_mean;
        }
        Ok(rt)
    }

    /// Forward pass. Returns (ms2 [L-1, n_ion] in (0,1), rt native, ccs native).
    pub fn forward(
        &self,
        pep: &Peptide,
        charge: i64,
        ms_shift: Option<&Array1<f32>>,
        chrom_shift: Option<&Array1<f32>>,
        chrom_affine: Option<(f32, f32)>,
    ) -> Result<(Array2<f32>, f32, f32)> {
        let encoded = self.encode(pep)?;
        let (ms2, ccs) = self.predict_charge(&encoded, charge, ms_shift)?;
        let rt = self.predict_rt(&encoded, chrom_shift, chrom_affine)?;
        Ok((ms2, rt, ccs))
    }
}
