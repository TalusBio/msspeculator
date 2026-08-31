//! Progress reporting for a library build.
//!
//! A build is long enough that a caller needs to know it is still moving, and its parts are
//! measured in different things: digestion consumes a file, prediction consumes the peptides that
//! came out of it. [`Phase`] is what lets one callback report all of them without a `done` that
//! means something different from one update to the next.

use std::cell::Cell;

/// Which part of a library build an update describes.
///
/// The phases run in this order and never interleave, so a caller can treat a phase change as
/// "the previous phase finished" without waiting for `done == total`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Reading the FASTA and enumerating tryptic peptides.
    Digesting,
    /// Reading the model. Announced rather than measured: loading an artifact is one call that
    /// reports nothing on the way through, so this exists to explain a pause that would
    /// otherwise look like a stall between the other two phases.
    Loading,
    /// Enumerating peptidoforms and predicting their spectra.
    Predicting,
}

impl Phase {
    /// A lowercase present participle, for a progress bar message or a log line.
    pub fn label(self) -> &'static str {
        match self {
            Self::Digesting => "digesting",
            Self::Loading => "loading",
            Self::Predicting => "predicting",
        }
    }

    /// Plural noun naming what `done` and `total` count, or empty for a phase that measures
    /// nothing.
    pub fn unit(self) -> &'static str {
        match self {
            Self::Digesting => "bytes",
            Self::Loading => "",
            Self::Predicting => "peptides",
        }
    }

    /// How well this phase knows what it reports.
    pub fn exactness(self) -> Exactness {
        match self {
            Self::Digesting => Exactness::Exact,
            // Nothing is counted, so nothing is exact.
            Self::Loading => Exactness::Approximate,
            // The producer enumerates ahead of the workers, bounded by the depth of the work
            // queue, so `done` leads what has actually been predicted by at most a few thousand
            // peptidoforms. Exact at the closing update, which lands after the writer finishes.
            Self::Predicting => Exactness::Approximate,
        }
    }
}

/// How much of what a phase claims is counted rather than estimated.
///
/// Carried so a renderer can decide how much to promise without matching on [`Phase`]: an
/// approximate count is one to show as a bar and not to quote as a number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exactness {
    /// `done` and `total` are both counted.
    Exact,
    /// `done` may run ahead of or behind the work that has actually finished, or `total` may be
    /// a projection. The magnitude is right; the last digits are not.
    Approximate,
}

/// How far a library build has got.
///
/// What the numbers count and how far to trust them are properties of the [`Phase`], reachable
/// through [`Phase::unit`] and [`Phase::exactness`]: storing them here too would be two fields
/// that can disagree with a third.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Progress {
    pub phase: Phase,
    /// Units completed. Monotone within a phase, and restarts from zero at a phase change.
    pub done: u64,
    /// The value `done` reaches when this phase ends, or `0` for a phase that measures nothing.
    /// Known before the phase starts, so an ETA is available from the first update.
    pub total: u64,
}

impl Progress {
    /// Completed fraction in `0.0..=1.0`, and `0.0` for a phase that measures nothing.
    pub fn fraction(self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.done as f64 / self.total as f64).clamp(0.0, 1.0)
        }
    }
}

/// What a caller hands to [`StreamOptions::progress`](crate::library::StreamOptions::progress).
///
/// `Send + Sync` because those bounds can be relaxed later without breaking a caller and cannot be
/// added later at all. The lifetime is spelled out because a bare trait-object alias would default
/// to `'static`, ruling out a closure that borrows a progress bar off the caller's stack.
pub type ProgressFn<'a> = dyn Fn(Progress) + Send + Sync + 'a;

/// One update per this many, so a proteome reports thousands of times rather than millions. The
/// exact value does not matter; that the pipeline does not pick it is what matters.
const REPORT_EVERY: u32 = 64;

/// The reporting end of a build: the one place that knows whether anyone is listening and how
/// often to speak.
///
/// The rate lives here rather than at the call sites, so the cadence is a decision instead of a
/// side effect of whatever the inference batch size happens to be.
pub(crate) struct Reporter<'a> {
    callback: Option<&'a ProgressFn<'a>>,
    /// Updates suppressed since the last one that went out. A `Cell`, not an atomic: every update
    /// comes from the producer thread.
    since: Cell<u32>,
}

impl<'a> Reporter<'a> {
    pub(crate) fn new(callback: Option<&'a ProgressFn<'a>>) -> Self {
        Self {
            callback,
            since: Cell::new(0),
        }
    }

    pub(crate) fn at(&self, phase: Phase, done: u64, total: u64) {
        let Some(callback) = self.callback else {
            return;
        };
        // Clamped rather than trusted: `done` counts what the pipeline consumed, and a FASTA that
        // grew between the metadata read and the last line would report past the end.
        let done = done.min(total);
        // A phase boundary always goes out. A bar that never reaches 100%, or starts at 40%
        // because the first updates were swallowed, is worse than one that moves in steps.
        let boundary = done == 0 || done >= total;
        let since = self.since.get() + 1;
        self.since.set(if boundary { 0 } else { since });
        if boundary || since.is_multiple_of(REPORT_EVERY) {
            callback(Progress { phase, done, total });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn progress(phase: Phase, done: u64, total: u64) -> Progress {
        Progress { phase, done, total }
    }

    #[test]
    fn fraction_is_bounded_and_survives_a_phase_with_no_size() {
        let at = |done, total| progress(Phase::Digesting, done, total).fraction();
        assert_eq!(at(0, 0), 0.0);
        assert_eq!(at(1, 4), 0.25);
        assert_eq!(at(4, 4), 1.0);
        assert_eq!(at(9, 4), 1.0);
    }

    #[test]
    fn every_phase_names_what_it_counts() {
        for phase in [Phase::Digesting, Phase::Loading, Phase::Predicting] {
            assert!(!phase.label().is_empty(), "{phase:?} has no label");
        }
        assert_eq!(Phase::Digesting.unit(), "bytes");
        assert_eq!(Phase::Digesting.exactness(), Exactness::Exact);
        assert_eq!(Phase::Predicting.exactness(), Exactness::Approximate);
        // Loading counts nothing, so it names nothing; the empty unit is what pairs with the
        // `total: 0` it reports.
        assert_eq!(Phase::Loading.unit(), "");
    }

    fn collect(updates: impl Fn(&Reporter<'_>)) -> Vec<Progress> {
        let seen = std::sync::Mutex::new(Vec::new());
        let callback = |progress: Progress| seen.lock().unwrap().push(progress);
        updates(&Reporter::new(Some(&callback)));
        seen.into_inner().unwrap()
    }

    #[test]
    fn a_reporter_without_a_callback_is_silent_and_one_with_it_clamps() {
        Reporter::new(None).at(Phase::Predicting, 5, 1);
        assert_eq!(collect(|r| r.at(Phase::Predicting, 5, 1))[0].done, 1);
    }

    /// The cadence is the reporter's decision, so it is the reporter's test: intermediate
    /// updates thin out, but neither end of a phase is ever swallowed.
    #[test]
    fn a_phase_reports_both_ends_and_thins_out_between_them() {
        let total = u64::from(REPORT_EVERY) * 4;
        let seen = collect(|reporter| {
            for done in 0..=total {
                reporter.at(Phase::Predicting, done, total);
            }
        });
        assert_eq!(seen.first().unwrap().done, 0);
        assert_eq!(seen.last().unwrap().done, total);
        assert!(seen.len() > 2, "no update between the ends: {}", seen.len());
        assert!(
            seen.len() < total as usize / 8,
            "barely thinned: {} of {total}",
            seen.len()
        );
        assert!(seen.windows(2).all(|w| w[0].done < w[1].done));
    }
}
