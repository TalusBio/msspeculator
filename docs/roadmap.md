# Roadmap

Updated 2026-08-11. The product milestone is a reproducible path from FASTA and prepared
experimental data to a Rust-generated spectral library that produces useful search results.
Validation is reported per dataset; a pooled score is not an acceptance criterion. ONNX is
deferred until it solves a measured deployment need.

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
- The full preparation config selects all non-test `prospect`, `tmt`, `multi_ptm`, and `tmt_ptm`
  archives. The separately labelled `test_ptm` record is excluded from training.

## Next work

1. **Complete and validate the full prepared catalog.** Run the global shard ranges, verify
   restart/skip behavior, finalize the manifest, and record row counts and dataset coverage.
2. **Train on the full non-test corpus.** Use the prepared manifest and run the five-preset sweep
   (`flash`, `small-2h`, `small`, `base-4h`, `base`), retaining per-dataset metrics and comparing
   the useful candidates on the real search fixture.
3. **Make library generation reproducible.** Write a manifest beside each generated library
   containing model identity, FASTA digest, digestion/modification settings, acquisition context,
   adapter version, and precursor/transition counts.
4. **Characterize mobility calibration.** Recheck the CCS-to-1/K0 residual on the larger model
   and data run before deciding whether calibration belongs in training or export.
5. **Close checkpoint/data-contract debt.** Serialize dataset names atomically with
   `ChromRunbook` rows, remove obsolete test-only real-data decode paths, and consolidate the
   duplicated RT-normalization implementations.
6. **Evaluate representation augmentation.** Compare the 1%-of-peptides chemistry-preserving
   substitution run against an unaugmented control before changing its rate or policy.
7. **Regenerate vendored UNIMOD assets reproducibly.** Refresh the generation logic from the
   upstream source and verify the checked-in result.
8. **Evaluate prepared-data curation.** Materialize chromatographic/identification QC and
   replicate-support statistics, then compare raw observations, apex/FWHM filtering, capped
   replicate weighting, and replicate-consensus targets without sacrificing singleton coverage
   by default.

## Parked

- ONNX export/runtime maintenance, until a concrete consumer requires it.
- Unspecific-window pretraining, until a non-tryptic target justifies the much larger stream.
- Additional search-engine adapters, until there is a named consumer and format contract.
- Further parallelism changes, until profiling of the full prepared-data/model workload shows a
  remaining bottleneck and the ordering/memory contract is explicit.
