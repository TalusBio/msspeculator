//! A temporary file that removes itself, shared by the crate's tests so the helper is written
//! once rather than once per module.

/// A scratch path that removes itself, so the tests need no dev-dependency for it.
pub(crate) struct Scratch(std::path::PathBuf);

impl Scratch {
    pub(crate) fn new(name: &str) -> Self {
        // Counted rather than timestamped: the tests run in parallel and several ask for the same
        // name, and two threads can read the same nanosecond. A collision means one test's `Drop`
        // deletes a file another is still reading.
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        Self(std::env::temp_dir().join(format!(
            "msspeculator-{}-{}-{name}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        )))
    }

    /// A scratch file that already holds `contents`.
    pub(crate) fn holding(name: &str, contents: &str) -> Self {
        let scratch = Self::new(name);
        std::fs::write(scratch.path(), contents).unwrap();
        scratch
    }

    pub(crate) fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
