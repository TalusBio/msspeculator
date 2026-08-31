//! The CLI's progress reporting, on top of the inference crate's callback.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::time::{Duration, Instant};

use msspeculator_inference::{Exactness, Phase, Progress};

/// Rewrite rate for the in-place line. Fast enough to look continuous, slow enough that a build
/// dispatching tens of thousands of batches does not spend its time on `write`.
const REDRAW: Duration = Duration::from_millis(100);

/// The phase a line shows before it has shown anything, so the first update reads as a change.
const NO_PHASE: u8 = u8::MAX;

/// A phase as an atomic-sized number. Hand-written rather than an `as` cast, because [`NO_PHASE`]
/// has to be a value no phase can take.
fn phase_index(phase: Phase) -> u8 {
    match phase {
        Phase::Digesting => 0,
        Phase::Loading => 1,
        Phase::Predicting => 2,
    }
}

/// Line rate for the appended form. A 70-second build leaves seven lines and a twelve-hour one
/// leaves about four thousand, which is a log a person can still scroll.
const LOG_INTERVAL: Duration = Duration::from_secs(10);

/// How the build reports itself.
#[derive(Clone, Copy)]
enum Style {
    /// One line rewritten in place, for a terminal.
    Live,
    /// A new line each interval, for a file. `\r` into a redirected stream produces one
    /// enormous line, so the two cases cannot share a renderer.
    Logged,
}

/// Restoring the terminal is the type's job, not the caller's.
///
/// `write_library` returns early on any failure, so an explicit call at the end of the happy path
/// leaves the live line on screen exactly when something has gone wrong — and anyhow then appends
/// `Error: ...` to the end of a half-drawn progress bar.
impl Drop for ProgressLine {
    fn drop(&mut self) {
        self.wipe();
    }
}

/// The CLI's progress reporter.
///
/// Reports through `&self` rather than `&mut self` because the callback is shared, not owned:
/// the throttle is an atomic instead of a lock, so a report that arrives while another is
/// rendering is dropped rather than queued. Dropping one is right for a status line; the next
/// one is along in milliseconds and carries a better number.
pub struct ProgressLine {
    /// `None` when reporting is off, which is the absence of a style rather than one of them.
    style: Option<Style>,
    started: Instant,
    /// Milliseconds since `started` at the last rendered update.
    rendered: AtomicU64,
    /// The phase of the last rendered update, or [`NO_PHASE`] before there was one. Kept so a
    /// phase change can bypass the throttle.
    phase: AtomicU8,
    /// Whether anything is currently on the terminal line, so the wipe knows to run.
    live: AtomicBool,
}

impl ProgressLine {
    /// Report to stderr, picking the style from what stderr turned out to be.
    pub fn for_stderr(enabled: bool) -> Self {
        let style = match (enabled, std::io::stderr().is_terminal()) {
            (false, _) => None,
            (true, true) => Some(Style::Live),
            (true, false) => Some(Style::Logged),
        };
        Self {
            style,
            started: Instant::now(),
            rendered: AtomicU64::new(0),
            phase: AtomicU8::new(NO_PHASE),
            live: AtomicBool::new(false),
        }
    }

    pub fn update(&self, progress: Progress) {
        let Some(style) = self.style else { return };
        let interval = match style {
            Style::Live => REDRAW,
            Style::Logged => LOG_INTERVAL,
        };
        let Some(now) = self.claim(progress.phase, interval) else {
            return;
        };
        let mut stderr = std::io::stderr().lock();
        let _ = match style {
            Style::Live => {
                self.live.store(true, Ordering::Relaxed);
                // Padded *and truncated* to a fixed width: padding stops a shorter line leaving
                // the tail of a longer one behind, and without the truncation a line that
                // overflows the width has exactly the problem the padding exists to prevent.
                write!(stderr, "\r{:<78.78}", self.render(progress, now))
            }
            Style::Logged => writeln!(stderr, "{}", self.render(progress, now)),
        };
        let _ = stderr.flush();
    }

    /// Whether this update gets to render, and the clock reading it renders with.
    ///
    /// Losing the exchange means another update is already rendering this instant; that one is
    /// dropped rather than queued, which is right for a status line. The next is along in
    /// milliseconds with a better number.
    fn claim(&self, phase: Phase, interval: Duration) -> Option<u64> {
        let now = self.started.elapsed().as_millis() as u64;
        // A phase change is never throttled. The interval exists to stop a fast phase rewriting
        // the line thousands of times a second, and dropping a change instead loses the label:
        // a build whose digestion finishes in nine milliseconds would show `digesting 100%` for
        // the whole of a four-minute model load, which is the stall `loading` exists to explain.
        // This also covers the first update of the build, which arrives too soon after `started`
        // to clear any interval.
        if self.phase.swap(phase_index(phase), Ordering::Relaxed) != phase_index(phase) {
            self.rendered.store(now, Ordering::Relaxed);
            return Some(now);
        }
        let previous = self.rendered.load(Ordering::Relaxed);
        if now.saturating_sub(previous) < interval.as_millis() as u64 {
            return None;
        }
        self.rendered
            .compare_exchange(previous, now, Ordering::Relaxed, Ordering::Relaxed)
            .ok()
            .map(|_| now)
    }

    /// Wipe the in-place line, so whatever the CLI prints next starts on a clean one.
    fn wipe(&self) {
        if self.live.swap(false, Ordering::Relaxed) {
            let mut stderr = std::io::stderr().lock();
            let _ = write!(stderr, "\r{:<78}\r", "");
            let _ = stderr.flush();
        }
    }

    fn render(&self, progress: Progress, elapsed_ms: u64) -> String {
        // A phase with no measurable size gets the label and the clock; a percentage of nothing
        // would be a number the build cannot back up.
        if progress.total == 0 {
            return format!(
                "{} ({} elapsed)",
                progress.phase.label(),
                duration(elapsed_ms)
            );
        }
        let approximately = match progress.phase.exactness() {
            Exactness::Exact => "",
            Exactness::Approximate => "~",
        };
        format!(
            "{} {:>3}% {}{}/{} {}  {} elapsed, {} left",
            progress.phase.label(),
            (progress.fraction() * 100.0) as u64,
            approximately,
            grouped(progress.done),
            grouped(progress.total),
            progress.phase.unit(),
            duration(elapsed_ms),
            remaining(progress, elapsed_ms),
        )
    }
}

/// The time left in the current phase, extrapolated from the rate so far.
///
/// Per phase rather than for the build, because the two phases are measured in different units
/// and run at unrelated rates: extrapolating one from the other would be arithmetic on numbers
/// that have nothing to do with each other.
fn remaining(progress: Progress, elapsed_ms: u64) -> String {
    if progress.done == 0 || progress.total == 0 {
        return "?".to_string();
    }
    let fraction = progress.fraction();
    if fraction <= 0.0 {
        return "?".to_string();
    }
    duration((elapsed_ms as f64 * (1.0 - fraction) / fraction) as u64)
}

fn duration(ms: u64) -> String {
    let seconds = ms / 1000;
    match (seconds / 3600, (seconds % 3600) / 60, seconds % 60) {
        (0, 0, s) => format!("{s}s"),
        (0, m, s) => format!("{m}m{s:02}s"),
        (h, m, _) => format!("{h}h{m:02}m"),
    }
}

/// Group thousands with underscores, the way Rust spells a large literal.
///
/// A progress line's whole job is being read at a glance, and `4200000` is not read at a glance.
fn grouped(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, digit) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push('_');
        }
        out.push(digit);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use msspeculator_inference::Phase;

    fn progress(done: u64, total: u64) -> Progress {
        Progress {
            phase: Phase::Predicting,
            done,
            total,
        }
    }

    #[test]
    fn thousands_are_grouped_at_every_size() {
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(999), "999");
        assert_eq!(grouped(1_000), "1_000");
        assert_eq!(grouped(4_200_000), "4_200_000");
    }

    #[test]
    fn durations_shorten_to_the_two_largest_units() {
        assert_eq!(duration(999), "0s");
        assert_eq!(duration(45_000), "45s");
        assert_eq!(duration(3_599_000), "59m59s");
        assert_eq!(duration(45_000_000), "12h30m");
    }

    #[test]
    fn an_eta_needs_progress_to_extrapolate_from() {
        assert_eq!(remaining(progress(0, 100), 5_000), "?");
        assert_eq!(remaining(progress(50, 0), 5_000), "?");
        // A quarter done after five seconds puts the rest fifteen seconds out.
        assert_eq!(remaining(progress(25, 100), 5_000), "15s");
    }

    #[test]
    fn a_line_names_its_unit_and_flags_an_approximate_count() {
        let line = ProgressLine::for_stderr(false);
        let rendered = line.render(progress(25, 100), 5_000);
        assert!(rendered.contains("peptides"), "{rendered}");
        assert!(rendered.contains("~25/100"), "{rendered}");
        assert!(rendered.starts_with("predicting  25%"), "{rendered}");
    }

    #[test]
    fn a_phase_with_no_measurable_size_shows_the_clock_and_no_percentage() {
        let line = ProgressLine::for_stderr(false);
        // The shape `Phase::Loading` actually reports, which is why this branch is live rather
        // than reachable only from a hand-built update.
        let loading = Progress {
            phase: Phase::Loading,
            done: 0,
            total: 0,
        };
        assert_eq!(line.render(loading, 5_000), "loading (5s elapsed)");
    }

    #[test]
    fn a_silent_line_renders_nothing_and_needs_no_terminal() {
        let line = ProgressLine::for_stderr(false);
        line.update(progress(1, 2));
        assert_eq!(line.phase.load(Ordering::Relaxed), NO_PHASE);
        assert!(!line.live.load(Ordering::Relaxed));
    }

    /// Both ends of a phase arrive within a millisecond of each other on a small build, so a
    /// throttle that applied to them would drop every update that names a phase but the first.
    #[test]
    fn a_phase_change_outranks_the_throttle() {
        let line = ProgressLine::for_stderr(false);
        assert!(
            line.claim(Phase::Digesting, REDRAW).is_some(),
            "the opening update was dropped"
        );
        assert!(
            line.claim(Phase::Digesting, REDRAW).is_none(),
            "the throttle stopped applying within a phase"
        );
        assert!(
            line.claim(Phase::Loading, REDRAW).is_some(),
            "a phase change was dropped, so its label never appeared"
        );
        assert!(
            line.claim(Phase::Predicting, REDRAW).is_some(),
            "a phase change was dropped, so its label never appeared"
        );
    }
}
