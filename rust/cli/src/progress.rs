//! The CLI's progress reporting, on top of the inference crate's callback.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use msspeculator_inference::{Exactness, Progress};

/// Rewrite rate for the in-place line. Fast enough to look continuous, slow enough that a build
/// dispatching tens of thousands of batches does not spend its time on `write`.
const REDRAW: Duration = Duration::from_millis(100);

/// Line rate for the appended form. A 70-second build leaves seven lines and a twelve-hour one
/// leaves about four thousand, which is a log a person can still scroll.
const LOG_INTERVAL: Duration = Duration::from_secs(10);

/// How the build reports itself.
enum Style {
    /// One line rewritten in place, for a terminal.
    Live,
    /// A new line each interval, for a file. `\r` into a redirected stream produces one
    /// enormous line, so the two cases cannot share a renderer.
    Logged,
    /// Nothing, for `--no-progress`.
    Silent,
}

/// The CLI's progress reporter.
///
/// Reports through `&self` rather than `&mut self` because the callback is shared, not owned:
/// the throttle is an atomic instead of a lock, so a report that arrives while another is
/// rendering is dropped rather than queued. Dropping one is right for a status line; the next
/// one is along in milliseconds and carries a better number.
pub struct ProgressLine {
    style: Style,
    started: Instant,
    /// Milliseconds since `started` at the last rendered update.
    rendered: AtomicU64,
    /// Whether anything is currently on the terminal line, so `finish` knows to wipe it.
    live: AtomicU64,
}

impl ProgressLine {
    /// Report to stderr, picking the style from what stderr turned out to be.
    pub fn for_stderr(enabled: bool) -> Self {
        let style = match (enabled, std::io::stderr().is_terminal()) {
            (false, _) => Style::Silent,
            (true, true) => Style::Live,
            (true, false) => Style::Logged,
        };
        Self {
            style,
            started: Instant::now(),
            rendered: AtomicU64::new(0),
            live: AtomicU64::new(0),
        }
    }

    pub fn update(&self, progress: Progress) {
        let interval = match self.style {
            Style::Silent => return,
            Style::Live => REDRAW,
            Style::Logged => LOG_INTERVAL,
        };
        let now = self.started.elapsed().as_millis() as u64;
        let previous = self.rendered.load(Ordering::Relaxed);
        if now.saturating_sub(previous) < interval.as_millis() as u64 {
            return;
        }
        // Losing the swap means another update is already rendering this instant; skip rather
        // than render twice.
        if self
            .rendered
            .compare_exchange(previous, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return;
        }
        let mut stderr = std::io::stderr().lock();
        let _ = match self.style {
            Style::Silent => return,
            Style::Live => {
                self.live.store(1, Ordering::Relaxed);
                // Padded to a fixed width so a shorter line cannot leave the tail of a longer
                // one behind it, which is the failure `\r` alone always has.
                write!(stderr, "\r{:<78}", self.render(progress, now))
            }
            Style::Logged => writeln!(stderr, "{}", self.render(progress, now)),
        };
        let _ = stderr.flush();
    }

    /// Wipe the in-place line, so whatever the CLI prints next starts on a clean one.
    pub fn finish(&self) {
        if self.live.swap(0, Ordering::Relaxed) == 1 {
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
        let approximately = match progress.exactness {
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
            progress.unit,
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
        Progress::new(Phase::Predicting, done, total)
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
        let rendered = line.render(Progress::new(Phase::Digesting, 0, 0), 5_000);
        assert_eq!(rendered, "digesting (5s elapsed)");
    }

    #[test]
    fn a_silent_line_renders_nothing_and_needs_no_terminal() {
        let line = ProgressLine::for_stderr(false);
        line.update(progress(1, 2));
        line.finish();
        assert_eq!(line.rendered.load(Ordering::Relaxed), 0);
    }
}
