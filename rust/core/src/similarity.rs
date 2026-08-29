//! Agreement between two spectra.
//!
//! Shared rather than private to the fitter, because two callers ask the same question of the
//! same weights: `fit_ms_context` optimizing a context against a library, and the doctor scoring
//! a model against the vendored reference panel. A doctor reaching into `fit` for its metric
//! would read as though it were fitting something.
//!
//! Takes iterators rather than a fixed container so an `ndarray` view and a flat `Vec` can both
//! reach it without one of them allocating a copy to look like the other.

/// Cosine similarity, accumulated in f64.
///
/// Zero when either spectrum has no intensity, which is a real state rather than an error: an
/// untrained model predicts nothing, and there is no angle between a vector and the origin.
pub fn cosine<A, B>(a: A, b: B) -> f32
where
    A: IntoIterator<Item = f32>,
    B: IntoIterator<Item = f32>,
{
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (x, y) in a.into_iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a <= 0.0 || norm_b <= 0.0 {
        return 0.0;
    }
    (dot / (norm_a.sqrt() * norm_b.sqrt())) as f32
}

/// Normalized spectral contrast angle in [0, 1], 1 = identical.
///
/// The reporting metric, matching `msspeculator.distill.losses.spectral_angle`. Both spectra are
/// read flat, so any grid works as long as the two agree on it.
pub fn spectral_angle<A, B>(a: A, b: B) -> f32
where
    A: IntoIterator<Item = f32>,
    B: IntoIterator<Item = f32>,
{
    let cos = cosine(a, b).clamp(-1.0, 1.0);
    1.0 - 2.0 * cos.acos() / std::f32::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;

    fn angle(a: &[f32], b: &[f32]) -> f32 {
        spectral_angle(a.iter().copied(), b.iter().copied())
    }

    #[test]
    fn identical_spectra_score_one() {
        let spectrum = [0.1, 1.0, 0.0, 0.4];
        assert!((angle(&spectrum, &spectrum) - 1.0).abs() < 1e-6);
    }

    /// Intensities are relative to a base peak on one side and raw model output on the other, so
    /// the metric has to ignore scale or nothing would ever agree.
    #[test]
    fn scale_does_not_matter() {
        let a = [0.1, 1.0, 0.0, 0.4];
        let b: Vec<f32> = a.iter().map(|value| value * 7.5).collect();
        assert!((angle(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn disjoint_spectra_score_zero() {
        assert!(angle(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn an_empty_prediction_scores_zero_rather_than_erroring() {
        assert_eq!(angle(&[0.0, 0.0], &[1.0, 0.5]), 0.0);
    }
}
