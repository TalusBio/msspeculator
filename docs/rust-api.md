# Rust API

Use `msspeculator-core` when a Rust program needs prediction results in memory. The
`msspeculator-cli` binary uses the same crate, but adds command-line parsing, FASTA digestion,
modification enumeration, and DIA-NN or mzSpecLib output.

## Add the dependency

The crate is not published yet. From a checkout, use a path dependency:

```toml
[dependencies]
msspeculator-core = { path = "../msspeculator/rust/core" }
anyhow = "1"
```

For a Git dependency, replace `path` with the repository URL and revision that contains the
Rust workspace:

```toml
msspeculator-core = { git = "https://github.com/jspaezp/distilltest.git", rev = "<commit>" }
```

Pin a commit for a service or application. The artifact format is checked at load time, but a
floating branch can still change prediction behavior between builds.

## Predict a peptide

`Artifact` loads a `.safetensors` export. `predict` parses the supported ProForma subset, runs the
model, and returns one `Prediction` containing precursor values and a struct-of-arrays fragment
table:

```rust,no_run
use anyhow::Result;
use msspeculator_core::{predict, Artifact};

fn main() -> Result<()> {
    let artifact = Artifact::load("model.safetensors")?;
    let result = predict(
        &artifact,
        "PEPC[UNIMOD:4]IDER",
        2,
        None,
        None,
        0.01,
    )?;

    println!(
        "{} z={} precursor_mz={} rt={} ccs={}",
        result.peptide, result.charge, result.precursor_mz, result.rt, result.ccs
    );
    for ((ion, ordinal), (mz, intensity)) in result
        .fragments
        .ion
        .iter()
        .zip(&result.fragments.ord)
        .zip(result.fragments.mz.iter().zip(&result.fragments.rel))
    {
        println!("{ion}{ordinal} m/z={mz} relative_intensity={intensity}");
    }
    Ok(())
}
```

The `min_intensity` argument is relative to the base peak. Set it to `0.0` to retain every
positive predicted fragment.

## Add acquisition context

Pass `MsContext::Factors` for instrument metadata and collision energy, or
`MsContext::Named` for a setup fitted into the artifact. A named chromatography context adjusts
retention time to that dataset's scale:

```rust,no_run
use msspeculator_core::{predict, Artifact, MsContext};

# fn run(artifact: &Artifact) -> anyhow::Result<()> {
let ms = MsContext::Factors {
    instrument: "Lumos".into(),
    detector: "FTMS".into(),
    fragmentation: "HCD".into(),
    energy: Some(30.0),
};
let result = predict(artifact, "PEPTIDER", 2, Some(&ms), Some("Evosep60SPD_heron"), 0.01)?;
# let _ = result;
# Ok(())
# }
```

The context name must exist in the artifact. `predict` returns an error for an unknown setup,
unsupported charge, malformed peptide, or an artifact format this build cannot read.

## Reuse work for batches

For repeated predictions with one context, resolve it once and use the prepared batch function:

```rust,no_run
use msspeculator_core::{predict_peptide_batch_charges_prepared, Artifact, MsContext,
                        PreparedContext};
use msspeculator_core::peptide::Peptide;

# fn run(artifact: &Artifact) -> anyhow::Result<()> {
let ms = MsContext::Named("Evosep60SPD_heron".into());
let context = PreparedContext::new(artifact, Some(&ms), None)?;
let peptides = vec![Peptide::parse("PEPTIDER")?, Peptide::parse("PEPTIDEK")?];
let predictions = predict_peptide_batch_charges_prepared(
    artifact,
    &peptides,
    &[2, 3],
    &context,
    0.01,
)?;
# let _ = predictions;
# Ok(())
# }
```

Batch inputs must share one sequence length. Grouping by length is the caller's responsibility.
The returned outer vector follows the input peptide order; each inner vector follows the
requested charge order.

## When to use the CLI

Use `msspeculator-cli` when you want the complete FASTA-to-library path without writing the
orchestration yourself:

```sh
cargo run --release -p msspeculator-cli -- \
  library --model model.safetensors --fasta proteome.fasta --out library.tsv
```

The CLI's FASTA digestion and writers are not part of the `msspeculator-core` API. Applications
that need a different input source or output format should use the core prediction functions and
own that small orchestration layer.
