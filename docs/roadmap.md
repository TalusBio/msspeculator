# Roadmap

Updated 2026-08-21. The product milestone is a reproducible path from FASTA and prepared
experimental data to a Rust-generated spectral library that produces useful search results.
Validation is reported per dataset; a pooled score is not an acceptance criterion. Production
inference is the Rust path via `export-rust`.

## Current baseline

- The `small` model has 132,358 parameters. Tanh-approximated GELU is the production default;
  exact GELU remains supported for compatibility and comparison.
- In the controlled 60-epoch comparison, weighted validation spectral angle was 0.7084 for
  tanh-GELU and 0.7075 for exact GELU. These historical pooled values summarize the experiment;
  new runs must report every dataset separately with its deduplicated validation count.
- On the real timsTOF fixture, exact GELU found 1,226 target precursors at 1% FDR and tanh-GELU
  found 1,158. Leaky ReLU was rejected after finding only 122.
- Rust generation of the 474,630-precursor / 16,650,774-transition fixture improved from
  168.62 seconds single-core to 11.53–12.97 seconds with equal-length batches, optimized matrix
  multiplication, bounded model workers, and one writer thread. The tanh-GELU artifact measured
  8.54 seconds in its controlled run.
- The DIA-NN TSV adapter loads in `timsseek`. The first real search completed in 194.6 seconds
  against `250225_Desnaux_200ng_Hela_ICC_on_DIA.d`, yielding 1,226 target precursors, 1,105
  unique modified sequences, and 365 FASTA proteins at 1% FDR.
- CCS is predicted by the model; the DIA-NN adapter converts it to Bruker 1/K0. Observed 1/K0
  was a median 3.47% below the library value, so calibration remains measurable work.

## Landed architecture

- Rust is the single source of truth for peptide chemistry, tokenization, tensor packing,
  safetensors inference, and DIA-NN library generation.
- Real-data validation is deduplicated and logged per dataset. Early stopping follows mean
  per-dataset spectral agreement and errors if an expected metric is missing.
- Real training logs progress and per-validation-check metrics, and saves `latest.ckpt` and
  `best.ckpt`.
  Pretraining uses OneCycle by default, logs learning rate, and saves `pretrain.ckpt` plus
  periodic warm-start snapshots. All durable artifacts can be mirrored incrementally to a
  per-run object-store prefix; checkpoints can be loaded directly from that prefix.
- Training can emit fixed-frame amino-acid, modification, and acquisition-context PCA panels,
  iRT calibration scatterplots, and teacher/student butterflies at low frequency. Snapshots are
  local artifacts, preserve their epoch paths in object storage, and are also sent to W&B.
- Real-data preparation is a separate range-addressable Polars ETL. A vendored Zenodo catalog
  and compressed shard index define the inputs; S3 is only an optional read-through cache.
- Every source shard produces one immutable prepared asset. Finalization verifies all shards and
  publishes one worker-independent training manifest.
- CPU training uses exact-length dense batches and direct Polars-to-tensor collation in the
  trainer process. The cloud training launcher requests 4 vCPUs and 16 GB: four model threads,
  no separate loader workers, and enough memory for the streamed one-shard-at-a-time workload.
- Training optionally applies chemistry-preserving residue substitutions to a configured
  fraction of peptides. The exact original-minus-replacement elemental composition and mass
  keep all targets invariant; production full-run configs currently select 1% of peptides.
- Real-data validation runs on a wall-clock cadence (hourly in full-run configs) rather than
  waiting for the end of multi-hour streaming epochs, plus one final check if needed. Early-stop
  patience counts checks.
- `ChromRunbook` owns its dataset name -> row map, so a corpus that gains a source cannot
  renumber rows that are already trained, and the map travels with the weights it indexes
  through the checkpoint and into the exported artifact.
- A published spectral library can be read, scored, and fitted against in Rust. `fit-context`
  gradient-descends one acquisition context row against a library with the backbone frozen,
  stores it in the artifact under a name, and `--ms-context NAME` then predicts with it. On the
  timsTOF heron library a fitted row moved held-out agreement 0.4691 -> 0.5316 where borrowing
  the closest existing instrument row reached only 0.5026. The Python `freeze_backbone` reference,
  re-run under the same project split, reached 0.4656 -> 0.5355, so the Rust fit recovers 89% of
  its improvement — the remaining gap is the finite-difference gradient and the padded fragment
  rows Python's grid carries, not a disagreement about the objective.
- The RT losses mask per row, so a source carrying only one of the two retention labels is
  supervised on what it has instead of being dropped from the corpus.
- The prepared catalog is complete and finalized: 5,174 shards, 21,003,479 rows, 41 datasets.
- One real-data entry point (`fit_realspeclib_datasets`, over any `BatchSource`) and one RT
  normalization (`establish_rt_norm`). The in-memory `RealLabels` fit path that carried a second
  copy of both was reachable only from its own tests and is gone.
- The vendored UNIMOD tables regenerate byte-identically from upstream: 40 nuclides and 1,560
  modifications, verified 2026-08-20. `unimod.tsv`'s mass column stays a test fixture — mass is
  computed from the composition and asserted against it on every row.
- Real-data training decays `lr` on the same plateau signal early stopping watches, cutting the
  rate before the run is allowed to end. A horizon-based schedule does not apply to this stage:
  it ends wherever early stopping lands, and the first full local run stopped at epoch 8 of a
  nominal 60. Off by default; the local full-run config cuts by half after 3 flat checks.
- The full preparation config selects all non-test `prospect`, `tmt`, `multi_ptm`, and `tmt_ptm`
  archives. The separately labelled `test_ptm` record is excluded from training.
- Library generation is reproducible and self-describing. One resolved configuration -- blake2b
  digests of the model and the FASTA, every digestion/modification/context/fragment setting as
  resolved, and the resulting counts -- is written beside the library as `config.json` and, for
  mzSpecLib output, into the library header itself, so a published library cannot be separated
  from its provenance by a copy. The format follows the `--out` suffix (`.mzspeclib.txt[.gz]`
  against everything else) rather than a flag that could contradict it. The mzSpecLib writer is
  ours, not `mzannotate`'s, because that crate re-derives masses from a `rustyms` peptidoform and
  would abort a run over a modification it cannot parse; the reference Python implementation reads
  the output back and reports no violations at any rule level.

## Next work

1. **Train on the full non-test corpus.** Use the prepared manifest and run the five-preset sweep
   (`flash`, `small-2h`, `small`, `base-4h`, `base`), retaining per-dataset metrics and comparing
   the useful candidates on the real search fixture.
2. **Characterize mobility calibration.** Recheck the CCS-to-1/K0 residual on the larger model
   and data run before deciding whether calibration belongs in training or export.
3. **Evaluate representation augmentation.** Compare the 1%-of-peptides chemistry-preserving
   substitution run against an unaugmented control before changing its rate or policy.
4. **Validate prepared-data curation across the full corpus.** The `v2` preparation path now
   estimates one robust peak width per raw file, centers it on each peptidoform apex across all
   charge/acquisition modes, requires four in-window PSMs, and retains the best two PSMs per
   charge/acquisition context. Compare its per-source retention and spectral-consistency reports
   with the unfiltered `v1` assets before making `v2` the training default. Replicate-consensus
   targets remain deferred.
5. **Use richer teacher supervision for modification pretraining.** Inventory what the parent
   model exposes for modified peptides, then deliberately sample supported modification/site
   combinations instead of relying mostly on incidental variable modifications. Measure
   modified-peptide spectral agreement separately by modification class and retain an
   unmodified control so improved PTM behavior cannot hide a base-peptide regression.
6. **Train on a corpus that includes a spectral library.** The reader, the per-row RT masking,
   and the named acquisition row are all in place; what is missing is the decision of whether a
   library enters as a prepared source or as extra shards, and threading `setup_id` through the
   batch so its row trains alongside the factor terms rather than only being fitted afterwards.

## Parked

- A portable export format (ONNX or otherwise), until a concrete consumer requires one. The ONNX
  path was removed on 2026-08-13; it had no consumer and its export baked in a sequence length.
- Unspecific-window pretraining, until a non-tryptic target justifies the much larger stream.
- Additional search-engine adapters, until there is a named consumer and format contract.
- Further parallelism changes, until profiling of the full prepared-data/model workload shows a
  remaining bottleneck and the ordering/memory contract is explicit.
