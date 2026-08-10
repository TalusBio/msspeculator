# Roadmap

Updated 2026-08-10. The product milestone is a reproducible path from FASTA and prepared
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
- Real training logs progress and per-epoch metrics, and saves `latest.ckpt` and `best.ckpt`.
  Pretraining uses OneCycle by default, logs learning rate, and saves `pretrain.ckpt`.
- Real-data preparation is a separate range-addressable Polars ETL. A vendored Zenodo catalog
  and compressed shard index define the inputs; S3 is only an optional read-through cache.
- Every source shard produces one immutable prepared asset. Finalization verifies all shards and
  publishes one worker-independent training manifest.
- The full preparation config selects all non-test `prospect`, `tmt`, `multi_ptm`, and `tmt_ptm`
  archives. The separately labelled `test_ptm` record is excluded from training.

## Next work

1. **Complete and validate the full prepared catalog.** Run the global shard ranges, verify
   restart/skip behavior, finalize the manifest, and record row counts and dataset coverage.
2. **Train on the full non-test corpus.** Use the prepared manifest, retain per-dataset metrics,
   and compare both `small` and `base` by validation strata and the real search fixture.
3. **Make library generation reproducible.** Write a manifest beside each generated library
   containing model identity, FASTA digest, digestion/modification settings, acquisition context,
   adapter version, and precursor/transition counts.
4. **Characterize mobility calibration.** Recheck the CCS-to-1/K0 residual on the larger model
   and data run before deciding whether calibration belongs in training or export.
5. **Close checkpoint/data-contract debt.** Serialize dataset names atomically with
   `ChromRunbook` rows, remove obsolete test-only real-data decode paths, and consolidate the
   duplicated RT-normalization implementations.
6. **Evaluate representation augmentation.** Add residue composition plus a compositional
   `Delta` modification variant before testing chemically equivalent residue/delta encodings.
7. **Regenerate vendored UNIMOD assets reproducibly.** Refresh the generation logic from the
   upstream source and verify the checked-in result.

## Parked

- ONNX export/runtime maintenance, until a concrete consumer requires it.
- Unspecific-window pretraining, until a non-tryptic target justifies the much larger stream.
- Additional search-engine adapters, until there is a named consumer and format contract.
- Further parallelism changes, until profiling of the full prepared-data/model workload shows a
  remaining bottleneck and the ordering/memory contract is explicit.
