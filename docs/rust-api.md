# Rust API

The workspace exposes two library crates:

- `msspeculator-core` loads artifacts and predicts peptides or batches.
- `msspeculator-inference` runs the production FASTA-to-library pipeline.

The CLI is a thin argument-parsing wrapper around the inference crate. An application can use the
same bounded queues, length buckets, worker threads, and writers without spawning a process.

## Add the crates

The crates are currently consumed from a checkout:

```toml
[dependencies]
msspeculator-core = { path = "../msspeculator/rust/core" }
msspeculator-inference = { path = "../msspeculator/rust/inference" }
anyhow = "1"
```

Pin a Git revision when building a service. The artifact reader checks its format version, but a
floating branch can still change prediction behavior.

## Load a model and predict

Built-in model selection is typed. External artifacts use `ModelSource::File`:

```rust,no_run
use anyhow::Result;
use msspeculator_core::{load_source, predict, BuiltinModel, ModelSource};

fn main() -> Result<()> {
    let loaded = load_source(ModelSource::Builtin(BuiltinModel::SmallV0))?;
    let result = predict(
        &loaded.artifact,
        "PEPC[UNIMOD:4]IDER",
        2,
        None,
        None,
        0.01,
    )?;
    println!("{} {}", result.peptide, result.precursor_mz);
    Ok(())
}
```

`predict` returns precursor values and a struct-of-arrays fragment table. `min_intensity` is
relative to the base peak. Set it to `0.0` to retain every positive fragment.

## Reuse context and batch work

Resolve an acquisition context once for repeated calls. Core batch inputs must share a sequence
length. Grouping by length is the caller's responsibility when using core directly.

```rust,no_run
use msspeculator_core::{
    predict_peptide_batch_charges_prepared, Artifact, MsContext, PreparedContext,
};
use msspeculator_core::peptide::Peptide;

fn run(artifact: &Artifact) -> anyhow::Result<()> {
    let ms = MsContext::Named("Evosep60SPD_heron".into());
    let context = PreparedContext::new(artifact, Some(&ms), None)?;
    let peptides = vec![Peptide::parse("PEPTIDER")?, Peptide::parse("PEPTIDEK")?];
    let predictions = predict_peptide_batch_charges_prepared(
        artifact, &peptides, &[2, 3], &context, 0.01,
    )?;
    println!("{}", predictions.len());
    Ok(())
}
```

`MsContext::Factors` accepts instrument, detector, fragmentation, and collision energy. A named
context must exist in the artifact. Unknown contexts, unsupported charges, malformed peptides,
and unreadable artifact versions return errors.

## Generate a library

Use `msspeculator-inference::write_library` for the complete optimized path. It reads FASTA,
digests and enumerates precursors, batches equal-length peptides, runs the bounded producer and
worker queues, and writes DIA-NN TSV or mzSpecLib output. `LibraryStats` reports counts for
proteins, peptides, precursors, fragments, and decoy precursors.

```rust,no_run
use std::path::Path;

use msspeculator_core::{BuiltinModel, ModelSource};
use msspeculator_inference::{write_library, LibraryOptions, StreamOptions};

fn run() -> anyhow::Result<()> {
    let fixed_mods = vec!["C[UNIMOD:4]".to_string()];
    let variable_mods = vec!["M[UNIMOD:35]".to_string()];
    let stats = write_library(&LibraryOptions {
        out: Path::new("library.tsv"),
        config_out: Some(Path::new("library.tsv.config.json")),
        stream: StreamOptions {
            model: ModelSource::Builtin(BuiltinModel::SmallV0),
            fasta: Path::new("proteome.fasta"),
            activation: None,
            ms_context: None,
            chrom_context: None,
            min_intensity: 0.01,
            missed_cleavages: 2,
            min_length: 7,
            max_length: 40,
            min_charge: 2,
            max_charge: 4,
            fixed_mods: &fixed_mods,
            variable_mods: &variable_mods,
            max_variable_mods: 1,
            max_fragments: None,
            generate_decoys: false,
        },
    })?;
    println!("{} precursors", stats.precursors);
    Ok(())
}
```

`LibraryOptions` uses the output suffix to choose the writer. `.mzspeclib.txt` and `.mzspeclib`
select mzSpecLib text; other suffixes select DIA-NN TSV. Add `.gz` to compress either stream.
Output order is unspecified because workers finish independently. The writer validates and caps
each precursor before serialization. The generated provenance records the package version and
source commit, along with the resolved model, input FASTA, and settings.

## Receive rows without writing a file

To keep the spectra in your own process instead, implement `LibrarySink` and call
`stream_library`, which takes `StreamOptions`: `LibraryOptions` minus the two fields that only
mean something when there is a file.

```rust,no_run
use msspeculator_inference::{stream_library, LibraryProvenance, LibrarySink, SpectrumRow};

struct Indexer {
    precursors: usize,
}

impl LibrarySink for Indexer {
    fn header(&mut self, provenance: &LibraryProvenance) -> anyhow::Result<()> {
        // The same provenance a written library carries, so an in-memory index can
        // record what produced it without a sidecar.
        println!("model {}", provenance.inputs.model);
        Ok(())
    }

    fn spectrum(&mut self, row: &SpectrumRow<'_>) -> anyhow::Result<()> {
        // `row` borrows from the prediction it came out of; copy anything you keep.
        self.precursors += 1;
        Ok(())
    }

    fn finish(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}
```

The sink is moved onto the writer thread while workers predict, which is why `LibrarySink`
requires `Send`. `provenance.output` is `None` here: there is no path, no suffix-chosen format,
and no compression to report.

Set `generate_decoys: true` to add pseudo-reversed decoys. Internal residues are reversed while
the first and last residues stay fixed, so `PEPTIDEK` becomes `PEDITPEK`. A decoy is skipped when
that stripped sequence is already a target or another decoy. DIA-NN rows use `Decoy=1` and a
`DECOY_` protein prefix. mzSpecLib entries claim a `Decoy` spectrum attribute set and use the
PSI-MS [`unnatural peptidoform decoy spectrum`](https://github.com/HUPO-PSI/psi-ms-CV/blob/master/psi-ms.obo)
origin term.
The `shuffle-and-reposition decoy spectrum` term is for rearranging peaks from an existing
spectrum, which this predicted-decoy path does not do.
For mzSpecLib output, each accepted target/decoy sequence pair shares a project-defined
`msspeculator:decoy_pair_id` attribute. Collision-skipped pairs retain the ID on the target only;
IDs are absent when decoys are off.

Use the CLI when its FASTA defaults and file formats are sufficient. Use the inference crate when
the application needs the same throughput with its own model source, paths, or surrounding
workflow. Use core when it owns batching and output itself.
