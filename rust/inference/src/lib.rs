//! High-throughput inference orchestration shared by the CLI and Rust applications.
//!
//! This crate owns FASTA digestion, modification enumeration, bounded producer/worker queues,
//! and output sinks. The model and peptide math remain in [`msspeculator_core`].

pub mod library;
pub mod mzspeclib;

pub use library::{write_library, LibraryOptions, LibraryStats};
pub use msspeculator_core::{
    Artifact, BuiltinModel, ModelSource, MsContext, Prediction, PreparedContext,
};
