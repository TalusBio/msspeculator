//! Progress reporting for a library build.
//!
//! A build is long enough that a caller needs to know it is still moving, and its parts are
//! measured in different things: digestion consumes a file, prediction consumes the peptides that
//! came out of it. [`Phase`] is what lets one callback report all of them without a `done` that
//! means something different from one update to the next.

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

    /// What this phase counts, and how well it knows it.
    fn measure(self) -> (&'static str, Exactness) {
        match self {
            Self::Digesting => ("bytes", Exactness::Exact),
            Self::Loading => ("models", Exactness::Exact),
            // The producer enumerates ahead of the workers, bounded by the depth of the work
            // queue, so `done` leads what has actually been predicted by at most a few thousand
            // peptidoforms. Exact at the closing update, which lands after the writer finishes.
            Self::Predicting => ("peptides", Exactness::Approximate),
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
/// `done` and `total` are always in the same unit, and `unit` names it, so a renderer can write a
/// line for a phase it has never heard of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Progress {
    pub phase: Phase,
    /// Units completed. Monotone within a phase, and restarts from zero at a phase change.
    pub done: u64,
    /// The value `done` reaches when this phase ends, or `0` for a phase with no measurable
    /// size. Known before the phase starts, so an ETA is available from the first update.
    pub total: u64,
    /// Plural noun naming what `done` and `total` count: `bytes`, `peptides`.
    pub unit: &'static str,
    pub exactness: Exactness,
}

impl Progress {
    /// Build an update, deriving the unit and the exactness from the phase.
    ///
    /// Public despite the struct's `#[non_exhaustive]`, because a downstream crate testing its
    /// own progress handling has to synthesize these. The attribute is there so a later field
    /// does not break that crate, not to stop it writing a test.
    pub fn new(phase: Phase, done: u64, total: u64) -> Self {
        let (unit, exactness) = phase.measure();
        Self {
            phase,
            done,
            total,
            unit,
            exactness,
        }
    }

    /// Completed fraction in `0.0..=1.0`, and `0.0` for a phase with no measurable size.
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
/// `Fn` rather than `FnMut` so a caller sharing state with it needs no wrapper of its own: an
/// `AtomicU64` captured by reference is enough. `Send + Sync` because those bounds can be relaxed
/// later without breaking a caller and cannot be added later at all; today every update comes
/// from the thread that called [`write_library`](crate::write_library).
///
/// The lifetime is spelled out because a bare trait-object alias would default to `'static`,
/// which would rule out the closure this is for: one borrowing a progress bar off the caller's
/// own stack.
pub type ProgressFn<'a> = dyn Fn(Progress) + Send + Sync + 'a;

/// The reporting end of a build, so the pipeline says `at(...)` instead of repeating the
/// `if let Some(callback)` that a build with no callback needs.
#[derive(Clone, Copy)]
pub(crate) struct Reporter<'a> {
    callback: Option<&'a ProgressFn<'a>>,
}

impl<'a> Reporter<'a> {
    pub(crate) fn new(callback: Option<&'a ProgressFn<'a>>) -> Self {
        Self { callback }
    }

    pub(crate) fn at(&self, phase: Phase, done: u64, total: u64) {
        if let Some(callback) = self.callback {
            // Clamped here rather than at every call site: `done` is counted from what the
            // pipeline consumed, and a FASTA that grew between the metadata read and the last
            // line would otherwise report past the end.
            callback(Progress::new(phase, done.min(total), total));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fraction_is_bounded_and_survives_a_phase_with_no_size() {
        let at = |done, total| Progress::new(Phase::Digesting, done, total).fraction();
        assert_eq!(at(0, 0), 0.0);
        assert_eq!(at(1, 4), 0.25);
        assert_eq!(at(4, 4), 1.0);
        assert_eq!(at(9, 4), 1.0);
    }

    #[test]
    fn every_phase_names_what_it_counts() {
        for phase in [Phase::Digesting, Phase::Loading, Phase::Predicting] {
            let progress = Progress::new(phase, 0, 1);
            assert!(!progress.unit.is_empty(), "{phase:?} has no unit");
            assert!(!phase.label().is_empty(), "{phase:?} has no label");
        }
        assert_eq!(Progress::new(Phase::Digesting, 0, 1).unit, "bytes");
        assert_eq!(
            Progress::new(Phase::Predicting, 0, 1).exactness,
            Exactness::Approximate
        );
    }

    #[test]
    fn a_reporter_without_a_callback_is_silent_and_one_with_it_clamps() {
        Reporter::new(None).at(Phase::Predicting, 5, 1);

        let seen = std::sync::Mutex::new(Vec::new());
        let callback = |progress: Progress| seen.lock().unwrap().push(progress);
        Reporter::new(Some(&callback)).at(Phase::Predicting, 5, 1);
        assert_eq!(seen.lock().unwrap()[0].done, 1);
    }
}
