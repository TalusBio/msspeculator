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
proteins, peptides, precursors, and fragments.

```rust,no_run
use msspeculator_core::{BuiltinModel, ModelSource};
use msspeculator_inference::{write_library, LibraryOptions};

fn run() -> anyhow::Result<()> {
    let fixed_mods = vec!["C[UNIMOD:4]".to_string()];
    let variable_mods = vec!["M[UNIMOD:35]".to_string()];
    let stats = write_library(&LibraryOptions {
        model: ModelSource::Builtin(BuiltinModel::SmallV0),
        fasta: "proteome.fasta",
        out: "library.tsv",
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
        config_out: Some("library.tsv.config.json"),
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

Use the CLI when its FASTA defaults and file formats are sufficient. Use the inference crate when
the application needs the same throughput with its own model source, paths, or surrounding
workflow. Use core when it owns batching and output itself.
