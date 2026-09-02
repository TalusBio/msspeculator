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
            progress: None,
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

## Check what a library was built from

A search reading a library someone else built can ask whether it matches the settings it is about
to search with. `check_library` compares the provenance the library carries against the `Settings`
these options resolve to.

```rust
use msspeculator_inference::{check_library, LibraryCheck};

// `None` for the sidecar looks beside the library, at `sidecar_path(&library_path)`.
match check_library(&library_path, None, &stream)? {
    LibraryCheck::Same => {}
    LibraryCheck::Different(differences) => {
        for difference in differences {
            eprintln!("{difference}");
        }
    }
    LibraryCheck::Unknown => eprintln!("no msspeculator provenance to compare against"),
}
```

Warn and proceed rather than refuse: a mismatch says the library answers a different question,
not that searching it is invalid. `Unknown` covers every library this cannot read a provenance
from, including a missing or unreadable one, and it is not a mismatch.

A library is read to the end of its header and no further, so it costs one open whatever its
size. A DIA-NN TSV has no header, so the sidecar is its only copy; pass its path when
`--config-out` put it somewhere other than beside the library. Neither side loads a model. The
expected side does hash the FASTA, which is a full read of it.

What is compared is exactly what `Settings` holds, and the question it answers is "would the same
library be generated", not "were the same arguments typed" — so a knob belongs in it when
changing that knob changes a byte of the output. A key only one side records is a difference,
since an unset knob is dropped from a header rather than written as a null: `--max-fragments`
dropped from a rebuild is as much a change as one given a new value. Equivalence across releases
is the one thing it cannot claim; `generator.commit` is recorded beside the settings for a reader
who needs that.

## Report progress

Set `progress` to watch a build that takes minutes. The callback receives a `Progress` carrying
the `Phase`, `done` and `total`; what those count and how far to trust them are properties of the
phase, through `Phase::unit` and `Phase::exactness`.

```rust,no_run
use msspeculator_inference::Progress;

fn watch(progress: Progress) {
    println!(
        "{}: {}/{} {}",
        progress.phase.label(),
        progress.done,
        progress.total,
        progress.phase.unit(),
    );
}
```

The phases run in order and never interleave. `Digesting` counts FASTA bytes consumed and
`Predicting` counts digested peptides; `Loading` reports `0/0`, because reading an artifact is a
single call that measures nothing, and a renderer should show its label alone. Prediction is
`Approximate` because the producer enumerates ahead of the workers by the depth of the work
queue; only the closing update lands after the last spectrum reaches the sink, so 100% means
written rather than queued.

Updates are bounded — thousands over a build, not millions, and both ends of every phase always
arrive, so a phase's first update has `done == 0` — but they are not tied to any clock. A callback
that moves a bar needs nothing else; one that writes a log line should throttle on its own.

Two things about drawing the bar are yours to get right, and neither is about the update rate.
A terminal and a redirected stream need two renderers rather than one narrowed: `\r` into a file
produces a single enormous line, so when the stream is not a terminal, trade the bar for one
appended line per interval. And close the bar — whatever draws it leaves the cursor on its own
line, so the next thing written continues it (`] 57/57 (0s)2 proteins -> ...`), and the next thing
written is often the error that ended the build. Close it on drop rather than at the end of the
happy path, which is the case that does not have it drawn. `msspeculator`'s own CLI does both in
`rust/cli/src/progress.rs`.

`LibraryStats` reports `digest`, `load` and `predict` as `Duration`s once the build finishes —
three disjoint numbers that sum to the build, because loading the model scales on the artifact and
the page cache rather than on the proteome or the precursor count. `write_library` records them in
the sidecar as `seconds_digesting`, `seconds_loading` and `seconds_predicting`.

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
        // `row.proteins` is a list, not a joined string: each `FastaId` displays with the
        // `DECOY_` prefix already applied when the spectrum is a decoy.
        for protein in row.proteins.iter() {
            println!("{protein}");
        }
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

A peptide from several proteins carries them all. `SpectrumRow.proteins` is a `ProteinGroup`,
which iterates `FastaId` values borrowed from the digest, so a peptide in ten proteins allocates
nothing. A `FastaId` is the first whitespace-delimited token of the FASTA description line,
unparsed: for a UniProt FASTA that is `sp|P00001|A_HUMAN`, which is a database, an accession and
an entry name run together, so it is deliberately not called an accession. Identifiers are
deduplicated, so a FASTA listing a protein twice under one name contributes one member, and a
group is ordered by identifier rather than by position in the file. DIA-NN output joins the group
with `;` because its format has one protein column; mzSpecLib writes one
`MS:1000885|protein accession` line per member, which is the form a reader can take apart again.

`SpectrumRow` likewise carries `peptide: &Peptide` and `stripped: Residues` rather than either
format's spelling of them: DIA-NN's writer renders `PEPC(UniMod:4)IDER`, mzSpecLib uses the
ProForma string in `proforma`.

Set `generate_decoys: true` to add pseudo-reversed decoys. Internal residues are reversed while
the first and last residues stay fixed, so `PEPTIDEK` becomes `PEDITPEK`. A decoy is skipped when
that stripped sequence is already a target; two decoys cannot collide with each other, because
reversing twice returns the original and the map is therefore injective. DIA-NN rows use `Decoy=1` and a
`DECOY_` protein prefix. mzSpecLib entries claim a `Decoy` spectrum attribute set and use the
PSI-MS [`unnatural peptidoform decoy spectrum`](https://github.com/HUPO-PSI/psi-ms-CV/blob/master/psi-ms.obo)
origin term.
The `shuffle-and-reposition decoy spectrum` term is for rearranging peaks from an existing
spectrum, which this predicted-decoy path does not do.
For mzSpecLib output, each accepted target/decoy precursor pair shares a project-defined
`msspeculator:decoy_pair_id` attribute: one ID per target peptidoform and charge, so a peptide's
modified forms and charge states are separate pairs. Collision-skipped pairs retain the ID on the
target only; IDs are absent when decoys are off.

Use the CLI when its FASTA defaults and file formats are sufficient. Use the inference crate when
the application needs the same throughput with its own model source, paths, or surrounding
workflow. Use core when it owns batching and output itself.
