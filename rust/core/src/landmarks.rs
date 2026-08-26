//! The two peptides that define the corpus retention index, and the check that a model is on it.
//!
//! The index a model reports is not a duration and not a published table: it is a linear
//! interpolation between two PROCAL standards pinned to 0 and 100. That convention is visible in
//! the source data; `TFAHTESHISK` carries exactly 0 and `SILDYVSLVEK` exactly 100 across
//! thousands of PSMs; which makes the scale definable rather than merely nameable, and makes
//! "is this artifact on it?" a question with a one-line answer.
//!
//! Deliberately not measured here: goodness of a linear fit against some external scale.
//! R-squared is invariant under any affine map of the predictions, so it cannot distinguish this
//! index from ten thousand times this index, and a reference table in another vendor's space can
//! only ever establish that the two are affine-related; never that ours is what it claims.
//! Predicting the anchors answers that directly.

use anyhow::Result;

use crate::artifact::Artifact;
use crate::model::Predictor;
use crate::peptide::Peptide;

/// `(sequence, index value it defines)`.
pub const ANCHORS: [(&str, f64); 2] = [("TFAHTESHISK", 0.0), ("SILDYVSLVEK", 100.0)];

/// How a library should describe its retention numbers.
pub const SCALE_DESCRIPTION: &str = "linear interpolation anchored at TFAHTESHISK = 0 and \
                                     SILDYVSLVEK = 100 (PROCAL standards, PROSPECT convention)";

/// Index units of slack allowed at an anchor.
///
/// Loose against the anchors themselves; a trained model lands within ~0.1; but far tighter
/// than the offset an artifact from another corpus would show, which is the case worth catching.
pub const ANCHOR_TOLERANCE: f64 = 2.0;

/// What an artifact predicts for each anchor.
#[derive(Debug, Clone)]
pub struct AnchorCheck {
    /// `(sequence, expected, predicted)`, in [`ANCHORS`] order.
    pub anchors: Vec<(&'static str, f64, f64)>,
    pub max_abs_error: f64,
}

impl AnchorCheck {
    pub fn on_scale(&self) -> bool {
        self.max_abs_error <= ANCHOR_TOLERANCE
    }
}

/// Predict the anchors with no acquisition context and report how far off they land.
///
/// Context-free because the index is what the model reports before any chromatography row is
/// applied; through a named dataset this would measure that dataset's gradient instead.
pub fn check_retention_scale(art: &Artifact) -> Result<AnchorCheck> {
    let predictor = Predictor::new(art);
    let mut anchors = Vec::with_capacity(ANCHORS.len());
    let mut max_abs_error = 0.0f64;
    for (sequence, expected) in ANCHORS {
        let peptide = Peptide::new(sequence.to_string(), Vec::new());
        let encoded = predictor.encode(&peptide)?;
        let predicted = f64::from(predictor.predict_rt(&encoded, None, None)?);
        if !predicted.is_finite() {
            anyhow::bail!("anchor {sequence} predicted a non-finite retention index");
        }
        max_abs_error = max_abs_error.max((predicted - expected).abs());
        anchors.push((sequence, expected, predicted));
    }
    Ok(AnchorCheck {
        anchors,
        max_abs_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_anchors_are_the_two_that_define_the_scale() {
        assert_eq!(ANCHORS.len(), 2);
        assert_eq!(ANCHORS[0].1, 0.0);
        assert_eq!(ANCHORS[1].1, 100.0);
        assert_ne!(ANCHORS[0].0, ANCHORS[1].0);
        for (sequence, _) in ANCHORS {
            assert!((7..=30).contains(&sequence.len()), "{sequence}");
            assert!(
                sequence
                    .chars()
                    .all(|aa| "GASPVTCLINDQKEMHFRYW".contains(aa)),
                "{sequence} has a residue the encoder does not know"
            );
            // Both anchors have to appear in `SCALE_DESCRIPTION`, or a library would describe a
            // convention different from the one it was checked against.
            assert!(SCALE_DESCRIPTION.contains(sequence), "{sequence}");
        }
    }

    #[test]
    fn the_bundled_model_is_on_the_corpus_scale() {
        // The positive case, which nothing could cover before weights were bundled: a randomly
        // initialised artifact can only ever demonstrate the check *failing*.
        let loaded = crate::builtin::load_model("builtin:small-v0").unwrap();
        let check = check_retention_scale(&loaded.artifact).unwrap();
        assert!(
            check.on_scale(),
            "bundled model is off its own scale by {}",
            check.max_abs_error
        );
        for (sequence, expected, predicted) in &check.anchors {
            assert!(
                (predicted - expected).abs() < 1.0,
                "{sequence} predicted {predicted}, defined as {expected}"
            );
        }
    }

    #[test]
    fn an_offset_beyond_tolerance_is_off_scale() {
        let on = AnchorCheck {
            anchors: vec![("A", 0.0, 0.11), ("B", 100.0, 100.11)],
            max_abs_error: 0.11,
        };
        assert!(on.on_scale());
        let off = AnchorCheck {
            anchors: vec![("A", 0.0, 25.7), ("B", 100.0, 102.9)],
            max_abs_error: 25.7,
        };
        assert!(!off.on_scale());
    }
}
