//! Fit one acquisition-context vector against a published spectral library.
//!
//! Answers "given a library with no recorded instrument or collision energy, what acquisition
//! context explains it. The fit optimizes the context directly rather than borrowing an existing
//! instrument's row. Everything else stays frozen: this fits sixteen-odd numbers, not a model.
//!
//! Two properties of the forward path make it cheap. [`Predictor::encode`] is charge- and
//! context-independent, so the backbone is paid once per library and cached. Context then enters
//! only at [`Predictor::predict_batch_charges`], which is two small linears. An objective
//! evaluation costs a head pass, not a model pass, and brute-forcing the gradient by finite
//! differences is affordable.
//!
//! There is no autodiff in this crate, and at this width there does not need to be: central
//! differences cost `2n + 1` head passes per step, 33 for a 16-d context. That stops being true
//! for a richer conditioner (per-layer shifts, low-rank adapters), which is the point at which an
//! analytic gradient or an autodiff dependency earns its keep. Finite differences would then
//! remain useful as the check that pins whatever replaces it.

use std::collections::BTreeMap;

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, Array3};

use crate::artifact::Artifact;
use crate::model::{EncodedPeptideBatch, Predictor};
use crate::speclib::LibraryPrecursor;
use crate::split::{assign_split, Split, SplitConfig};

/// How hard to fit. The defaults are what the Python reference converged with.
#[derive(Debug, Clone)]
pub struct FitConfig {
    /// Maximum passes over the training precursors. Held-out agreement is checked after each, and
    /// the best-scoring context is what gets returned, so this is a ceiling rather than a target.
    pub epochs: usize,
    /// Stop after this many epochs without a better held-out score. The reference plateaued after
    /// one pass, so a couple of epochs of patience is enough to catch a slower library without
    /// paying for a dozen that change nothing.
    pub patience: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    /// Central-difference probe. Large enough to clear f32 noise in the objective, small enough
    /// that the secant still tracks the tangent.
    pub probe: f32,
    /// The project's split, not a fit-local one: the corpus and the library have to agree about
    /// which peptides the model was allowed to see.
    pub split: SplitConfig,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            epochs: 12,
            patience: 2,
            batch_size: 256,
            learning_rate: 0.05,
            probe: 1e-3,
            split: SplitConfig::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FitReport {
    pub context: Vec<f32>,
    pub context_dim: usize,
    pub train: usize,
    pub val: usize,
    /// Counted and never touched, so the report says what it left alone.
    pub test: usize,
    pub spectral_angle_before: f32,
    pub spectral_angle_after: f32,
    /// Training objective after each epoch, over the whole training set so the entries are
    /// comparable to each other. A fit that went nowhere is visible here rather than having to be
    /// inferred from the final number.
    pub objective: Vec<f32>,
    /// Held-out agreement after each epoch. The reported context is the best of these, not the
    /// last, so this also shows how much was left on the table by stopping where it stopped.
    pub held_out: Vec<f32>,
}

/// One same-length, same-charge group: the backbone output, and the library's grid beside it.
struct Group {
    encoded: EncodedPeptideBatch,
    charge: i64,
    /// `[precursor, site, ion]`, on the same grid `Predictor` emits.
    observed: Array3<f32>,
}

/// Cosine similarity between two flattened grids. Both dense: a library reports about a dozen of
/// the fifty-odd cells and the rest are taken at face value as near-zero, which is what the
/// intensity floor measured on real files supports.
fn cosine(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    crate::similarity::cosine(a.iter().copied(), b.iter().copied())
}

/// Normalized spectral contrast angle in [0, 1], 1 = identical.
fn spectral_angle(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    crate::similarity::spectral_angle(a.iter().copied(), b.iter().copied())
}

/// Build the encoded groups for one split.
fn groups_for(
    predictor: &Predictor<'_>,
    precursors: &[LibraryPrecursor],
    wanted: &[usize],
    batch_size: usize,
) -> Result<Vec<Group>> {
    // Grouped by (length, charge): the batch encoder needs one shared length, and one charge per
    // group keeps the charge heads from computing charges no precursor in it asked for.
    let mut buckets: BTreeMap<(usize, i64), Vec<usize>> = BTreeMap::new();
    for &index in wanted {
        let precursor = &precursors[index];
        buckets
            .entry((precursor.peptide.length(), precursor.charge as i64))
            .or_default()
            .push(index);
    }
    let mut groups = Vec::new();
    for ((length, charge), members) in buckets {
        if length < 2 {
            continue;
        }
        for chunk in members.chunks(batch_size) {
            let peptides: Vec<_> = chunk
                .iter()
                .map(|&index| precursors[index].peptide.clone())
                .collect();
            let encoded = predictor.encode_batch(&peptides)?;
            let sites = length - 1;
            let n_ion = crate::chem::ION_TYPES.len();
            let mut observed = Array3::<f32>::zeros((chunk.len(), sites, n_ion));
            for (row, &index) in chunk.iter().enumerate() {
                for fragment in &precursors[index].fragments {
                    let site = fragment.site as usize;
                    let ion = fragment.ion as usize;
                    if site < sites && ion < n_ion {
                        observed[[row, site, ion]] = fragment.intensity;
                    }
                }
            }
            groups.push(Group {
                encoded,
                charge,
                observed,
            });
        }
    }
    Ok(groups)
}

/// Mean `1 - cosine` over a set of groups, with `context` applied.
fn objective(predictor: &Predictor<'_>, groups: &[Group], context: &Array1<f32>) -> Result<f32> {
    let shift = predictor.context_shift(context.view())?;
    let mut total = 0.0f64;
    let mut count = 0usize;
    for group in groups {
        let outputs =
            predictor.predict_batch_charges(&group.encoded, &[group.charge], Some(&shift))?;
        for (row, charge_outputs) in outputs.iter().enumerate() {
            let (predicted, _ccs) = charge_outputs
                .first()
                .ok_or_else(|| anyhow!("one requested charge yields one output"))?;
            let truth = group.observed.index_axis(ndarray::Axis(0), row).to_owned();
            total += 1.0 - cosine(predicted, &truth) as f64;
            count += 1;
        }
    }
    if count == 0 {
        return Err(anyhow!("no precursors to fit against"));
    }
    Ok((total / count as f64) as f32)
}

/// Median spectral angle over a set of groups, with `context` applied.
fn median_spectral_angle(
    predictor: &Predictor<'_>,
    groups: &[Group],
    context: &Array1<f32>,
) -> Result<f32> {
    let shift = predictor.context_shift(context.view())?;
    let mut angles: Vec<f32> = Vec::new();
    for group in groups {
        let outputs =
            predictor.predict_batch_charges(&group.encoded, &[group.charge], Some(&shift))?;
        for (row, charge_outputs) in outputs.iter().enumerate() {
            let (predicted, _ccs) = charge_outputs
                .first()
                .ok_or_else(|| anyhow!("one requested charge yields one output"))?;
            let truth = group.observed.index_axis(ndarray::Axis(0), row).to_owned();
            angles.push(spectral_angle(predicted, &truth));
        }
    }
    if angles.is_empty() {
        return Err(anyhow!("no precursors to score"));
    }
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(angles[angles.len() / 2])
}

/// Adam, kept local because it exists only to move sixteen numbers.
struct Adam {
    m: Array1<f32>,
    v: Array1<f32>,
    step: i32,
}

impl Adam {
    fn new(width: usize) -> Self {
        Self {
            m: Array1::zeros(width),
            v: Array1::zeros(width),
            step: 0,
        }
    }

    fn apply(&mut self, context: &mut Array1<f32>, gradient: &Array1<f32>, learning_rate: f32) {
        const BETA1: f32 = 0.9;
        const BETA2: f32 = 0.999;
        const EPS: f32 = 1e-8;
        self.step += 1;
        let bias1 = 1.0 - BETA1.powi(self.step);
        let bias2 = 1.0 - BETA2.powi(self.step);
        for i in 0..context.len() {
            self.m[i] = BETA1 * self.m[i] + (1.0 - BETA1) * gradient[i];
            self.v[i] = BETA2 * self.v[i] + (1.0 - BETA2) * gradient[i] * gradient[i];
            let m_hat = self.m[i] / bias1;
            let v_hat = self.v[i] / bias2;
            context[i] -= learning_rate * m_hat / (v_hat.sqrt() + EPS);
        }
    }
}

/// Central-difference gradient of the objective over one group set.
fn finite_difference(
    predictor: &Predictor<'_>,
    groups: &[Group],
    context: &Array1<f32>,
    probe: f32,
) -> Result<Array1<f32>> {
    let mut gradient = Array1::zeros(context.len());
    for i in 0..context.len() {
        let mut plus = context.clone();
        let mut minus = context.clone();
        plus[i] += probe;
        minus[i] -= probe;
        let up = objective(predictor, groups, &plus)?;
        let down = objective(predictor, groups, &minus)?;
        gradient[i] = (up - down) / (2.0 * probe);
    }
    Ok(gradient)
}

/// Fit an acquisition context against a library, reporting on peptides it never fitted on.
pub fn fit_ms_context(
    art: &Artifact,
    precursors: &[LibraryPrecursor],
    config: &FitConfig,
) -> Result<FitReport> {
    let predictor = Predictor::new(art);
    let context_dim = art.meta.config.context_dim;
    if context_dim == 0 {
        return Err(anyhow!(
            "this artifact has context_dim 0, so it has no acquisition context to fit"
        ));
    }

    let mut train = Vec::new();
    let mut val = Vec::new();
    let mut test = 0usize;
    for (index, precursor) in precursors.iter().enumerate() {
        // On the stripped sequence, so every modified form and charge of one peptide travels
        // together and a fitted context is never reported on a peptide it was fitted on.
        match assign_split(&precursor.peptide.sequence, &config.split) {
            Split::Train => train.push(index),
            Split::Val => val.push(index),
            Split::Test => test += 1,
        }
    }
    if train.is_empty() || val.is_empty() {
        return Err(anyhow!(
            "library splits to {} train and {} val precursors; too few to fit and report",
            train.len(),
            val.len()
        ));
    }

    let train_groups = groups_for(&predictor, precursors, &train, config.batch_size)?;
    let val_groups = groups_for(&predictor, precursors, &val, config.batch_size)?;

    // Zero is the honest starting point: it is the context-free model, so the "before" number is
    // what the checkpoint predicts knowing nothing about this run.
    let mut context = Array1::<f32>::zeros(context_dim);
    let before = median_spectral_angle(&predictor, &val_groups, &context)?;

    let mut adam = Adam::new(context_dim);
    let mut objective_trace = Vec::new();
    let mut held_out_trace = Vec::new();
    // Kept separately from the live context: more passes can and do walk past the best held-out
    // score, and returning where the loop happened to stop would report a context nobody chose.
    let mut best_context = context.clone();
    let mut best_angle = before;
    let mut stale = 0usize;
    for _ in 0..config.epochs {
        // Gradients accumulate across groups until `batch_size` precursors have contributed.
        // Groups are keyed by (length, charge) so the encoder and the charge heads each see one
        // shape, which leaves a long tail of small ones, and Adam normalizes per coordinate, so
        // stepping on a six-precursor group moves as far as stepping on a full one, just in a
        // noisier direction. Accumulating first makes every step's sample size the configured one
        // regardless of how a particular library happens to bucket.
        let mut accumulated: Array1<f32> = Array1::zeros(context_dim);
        let mut seen = 0usize;
        for group in &train_groups {
            let rows = group.observed.shape()[0];
            let batch = std::slice::from_ref(group);
            let gradient = finite_difference(&predictor, batch, &context, config.probe)?;
            accumulated = accumulated + gradient * rows as f32;
            seen += rows;
            if seen >= config.batch_size {
                adam.apply(
                    &mut context,
                    &(&accumulated / seen as f32),
                    config.learning_rate,
                );
                accumulated = Array1::zeros(context_dim);
                seen = 0;
            }
        }
        if seen > 0 {
            adam.apply(
                &mut context,
                &(&accumulated / seen as f32),
                config.learning_rate,
            );
        }
        // Once per epoch over the whole training set, so the trace is comparable across steps.
        // A per-group loss would compare different peptide lengths to each other and read as
        // noise whether or not the fit was working.
        objective_trace.push(objective(&predictor, &train_groups, &context)?);

        let angle = median_spectral_angle(&predictor, &val_groups, &context)?;
        held_out_trace.push(angle);
        if angle > best_angle {
            best_angle = angle;
            best_context = context.clone();
            stale = 0;
        } else {
            stale += 1;
            if stale >= config.patience {
                break;
            }
        }
    }
    let after = best_angle;
    let context = best_context;

    Ok(FitReport {
        context: context.to_vec(),
        context_dim,
        train: train.len(),
        val: val.len(),
        test,
        spectral_angle_before: before,
        spectral_angle_after: after,
        objective: objective_trace,
        held_out: held_out_trace,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn cosine_and_angle_agree_at_the_extremes() {
        let a = array![[1.0f32, 0.0], [0.5, 0.25]];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
        assert!((spectral_angle(&a, &a) - 1.0).abs() < 1e-5);
        let orthogonal = array![[0.0f32, 1.0], [0.0, 0.0]];
        let b = array![[1.0f32, 0.0], [0.0, 0.0]];
        assert!(cosine(&b, &orthogonal).abs() < 1e-6);
        assert!(spectral_angle(&b, &orthogonal).abs() < 1e-5);
        // An all-zero side has no direction, so there is nothing to be similar to.
        let zero = Array2::<f32>::zeros((2, 2));
        assert_eq!(cosine(&a, &zero), 0.0);
    }

    #[test]
    fn adam_descends_a_quadratic_to_its_optimum() {
        // The optimizer on its own, against a bowl whose minimum is known. Finite differences are
        // exact for a quadratic, so this pins the step rule rather than the probe.
        let target = array![1.5f32, -2.0, 0.25];
        let mut x = Array1::<f32>::zeros(3);
        let mut adam = Adam::new(3);
        for _ in 0..2000 {
            let gradient = (&x - &target).mapv(|v| 2.0 * v);
            adam.apply(&mut x, &gradient, 0.05);
        }
        for i in 0..3 {
            assert!(
                (x[i] - target[i]).abs() < 1e-2,
                "dim {i}: {} != {}",
                x[i],
                target[i]
            );
        }
    }

    #[test]
    fn central_differences_match_a_closed_form_gradient() {
        // f(x) = sum((x - c)^2), grad = 2(x - c). Same probe the fit uses.
        let c = array![0.5f32, -1.0];
        let f = |x: &Array1<f32>| -> f32 { (x - &c).mapv(|v| v * v).sum() };
        let x = array![0.1f32, 0.4];
        let probe = 1e-3f32;
        for i in 0..2 {
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[i] += probe;
            minus[i] -= probe;
            let numeric = (f(&plus) - f(&minus)) / (2.0 * probe);
            let exact = 2.0 * (x[i] - c[i]);
            assert!(
                (numeric - exact).abs() < 1e-3,
                "dim {i}: {numeric} != {exact}"
            );
        }
    }
}
