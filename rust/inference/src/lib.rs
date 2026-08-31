//! High-throughput inference orchestration shared by the CLI and Rust applications.
//!
//! This crate owns FASTA digestion, modification enumeration, bounded producer/worker queues,
//! and output sinks. The model and peptide math remain in [`msspeculator_core`].

pub mod library;
pub mod mzspeclib;
pub mod progress;
mod proteome;
pub mod provenance;

pub use library::{
    stream_library, write_library, LibraryOptions, LibrarySink, LibraryStats, Peak, SpectrumRow,
    StreamOptions,
};
pub use msspeculator_core::{
    Artifact, BuiltinModel, ModelSource, MsContext, Prediction, PreparedContext,
};
pub use progress::{Exactness, Phase, Progress, ProgressFn};
pub use proteome::{FastaId, ProteinGroup, Residues};
pub use provenance::LibraryProvenance;
