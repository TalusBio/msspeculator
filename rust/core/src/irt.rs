//! Least-squares agreement between observed and predicted retention indices.
//!
//! Lives in core rather than beside the CLI's plotting because two callers need the same numbers:
//! `msspeculator-cli doctor`, which reports them for a set of portable weights, and the Python
//! training panel, which reads them through the pyo3 seam every snapshot. A slope that differs
//! between the two would make "is the RT scale right?" unanswerable, because the answer would
//! depend on which side asked.
//!
//! The math only; rendering stays with whoever is drawing.

/// Ordinary-least-squares fit of predicted retention index against observed, plus the residual
/// summary a reader actually judges the fit by.
///
/// `slope` and `r_squared` are 0.0 for a degenerate input (no spread in the observed values),
/// which is what an untrained model produces: a flat prediction has no line to fit.
#[derive(Debug, Clone, PartialEq)]
pub struct IrtSummary {
    pub n: usize,
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub mae: f64,
}

/// Summarize `predicted` against `observed`, pairwise.
///
/// Callers are responsible for the inputs being the same length and non-empty; an empty input
/// divides by zero and yields NaN rather than an error, which is the behaviour every existing
/// caller was written against.
pub fn summarize(observed: &[f64], predicted: &[f64]) -> IrtSummary {
    let n = observed.len();
    let mean_x = observed.iter().sum::<f64>() / n as f64;
    let mean_y = predicted.iter().sum::<f64>() / n as f64;
    let covariance = observed
        .iter()
        .zip(predicted)
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>();
    let variance_x = observed.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>();
    let variance_y = predicted.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>();
    let slope = if variance_x > 0.0 {
        covariance / variance_x
    } else {
        0.0
    };
    let intercept = mean_y - slope * mean_x;
    let r_squared = if variance_x > 0.0 && variance_y > 0.0 {
        covariance.powi(2) / (variance_x * variance_y)
    } else {
        0.0
    };
    let mae = observed
        .iter()
        .zip(predicted)
        .map(|(x, y)| (x - y).abs())
        .sum::<f64>()
        / n as f64;
    IrtSummary {
        n,
        slope,
        intercept,
        r_squared,
        mae,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "{actual} != {expected} within 1e-9"
        );
    }

    #[test]
    fn identity_is_a_perfect_fit() {
        let observed = [-24.9, 0.0, 12.4, 42.3, 100.0];
        let summary = summarize(&observed, &observed);
        assert_eq!(summary.n, 5);
        close(summary.slope, 1.0);
        close(summary.intercept, 0.0);
        close(summary.r_squared, 1.0);
        close(summary.mae, 0.0);
    }

    #[test]
    fn recovers_a_known_affine() {
        let observed = [0.0, 10.0, 20.0, 30.0];
        // predicted = 2 * observed + 5, exactly.
        let predicted: Vec<f64> = observed.iter().map(|x| 2.0 * x + 5.0).collect();
        let summary = summarize(&observed, &predicted);
        close(summary.slope, 2.0);
        close(summary.intercept, 5.0);
        close(summary.r_squared, 1.0);
    }

    /// An untrained model predicts one number for everything. That is a real state the doctor has
    /// to report rather than divide by zero on, and 0.0 is what it reports.
    #[test]
    fn a_flat_prediction_has_no_slope() {
        let observed = [0.0, 10.0, 20.0];
        let summary = summarize(&observed, &[7.0, 7.0, 7.0]);
        close(summary.slope, 0.0);
        close(summary.r_squared, 0.0);
        close(summary.intercept, 7.0);
    }

    /// No spread in the observed values either; both guards fire at once.
    #[test]
    fn a_degenerate_panel_has_no_slope() {
        let summary = summarize(&[5.0, 5.0], &[1.0, 3.0]);
        close(summary.slope, 0.0);
        close(summary.r_squared, 0.0);
    }

    #[test]
    fn mae_is_the_mean_absolute_residual() {
        let summary = summarize(&[0.0, 10.0], &[1.0, 7.0]);
        close(summary.mae, 2.0);
    }
}
