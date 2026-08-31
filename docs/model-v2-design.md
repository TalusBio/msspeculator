# Student model v2: factor-conditioned heads (as built)

The student composes acquisition/chromatography conditioning from **named factors** with a
single neutral convention (blank = zero). This replaced the old CE-fallback ladder
(`ce_center`/`ce_scale`/`resolve_ce`) and two-table `ContextBook`. Context is never fabricated:
a missing factor contributes the zero/neutral value.

## Conditioning routing

| Head | Peptide repr | Charge | MS Context | Chrom Runbook |
|------|:---:|:---:|:---:|:---:|
| Fragments `(N,S,W)` | per-residue | yes | yes | no |
| Mobility / CCS `(N,1)` | pooled | yes | no | no |
| Retention / RT `(N,1)` | pooled | no | no | yes |

CCS is single-point for now. Mobility distributions are deferred. CCS uses only the peptide and
charge, not MS context.

Beside the factors, `MSContextEncoder` carries a table of **named acquisition setups**: rows
addressed by name, additive and zero-init like the factor terms. A source that records no factors
has nothing for them to compose from. A published library may report no instrument or collision
energy. A timsTOF may ramp energy with ion mobility. In either case, its offset from the base model is fitted
as a row instead (`msspeculator-cli fit-context --save-as NAME`) and addressed with
`--ms-context NAME`. This does not weaken the neutral convention: a source nobody named uses row
0 and changes nothing.

## Module tree (shapes; N batch, S residues, E model dim, W fragment channels)

```
tokens (peptide + mods) ── Backbone ──▶ PeptideRepr (N,S,E)
                                          ├─ per-residue ─────────────────┐
                                          └─ MaskedMeanPool ─▶ Pooled (N,E)

charge (N)        ── ChargeEmb ─────────▶ (N,E)

MS factors:  instrument, detector, fragmentation_type ── Emb each ─┐
             fragmentation_energy ── MLP (Linear=learned affine) ──┴─ Σ ─▶ MSContext (N,E)
                                                                        [all factors absent → zeros]

dataset (N)  ── ChromRunbook (Emb, row 0 = iRT-neutral) ────────────────▶ (N,E)
                 └─ per-dataset exp(log_scale), shift ──────────────────▶ raw-RT affine

Fragments:  (PeptideRepr ⊕ broadcast MSContext ⊕ broadcast ChargeEmb)
            ─▶ FragRepr (N,S,E) ─▶ [FragBackbone: identity in v1] ─▶ Decoder (Linear E→W) ─▶ MS2 (N,S,W)
Mobility:   (Pooled ⊕ ChargeEmb)                 ─▶ MLP ─▶ CCS (N,1)
Retention:  (Pooled ⊕ ChromRunbook[dataset])     ─▶ MLP ─▶ base RT
             base RT × exp(log_scale[dataset]) + shift[dataset] ───────▶ raw RT (N,1)
```

`⊕` = elementwise sum (every term is dim E; the zero vector is a strict no-op, so blank context
= base prediction). Broadcast expands `(N,E)` to `(N,S,E)` for the per-residue fragment path.

## Factors

- **MS Context** (MS2 side): `instrument`, `detector`, and `fragmentation_type` are categorical;
  each has an embedding with index 0 = "unknown"/blank (zero row). `fragmentation_energy` is a continuous NCE
  fed straight to an `MLP` (`Linear(1,E)→GELU→Linear`); the first `Linear` is the learned affine
  and uses a learned normalization, not a fixed center (replaces ce_center=30/ce_scale=10). No BatchNorm:
  real runs share one NCE per raw_file, so a batch is routinely single-valued and BatchNorm's
  variance normalization would collapse it. Sum of the four → MSContext. Every factor omitted →
  zeros → base.
- **Charge**: embedding, index by precursor charge. Feeds fragments + mobility.
- **Chrom Runbook** (RT side): an embedding plus output `log_scale` and `shift`, all keyed by
  **dataset**, with `row 0 = iRT / neutral`. Predict iRT through row 0; predict raw RT through the
  dataset row and its positive `exp(log_scale)` affine. Where a dataset carries both iRT and raw
  RT, train both targets. Zero initialization makes the initial affine the identity.

## Neutral convention

One rule everywhere: **the zero vector is the context-free base.** Blank MS Context = zeros;
Runbook row 0 = zeros (until trained); an unknown categorical = its index-0 zero row. No scalar
defaults, no `resolve_ce`, no fabricated NCE.

## Landed implementation map

- `models/context.py` implements `MSContextEncoder` and `ChromRunbook`; there is no CE fallback or
  two-table context book.
- `models/student.py` routes charge, MS context, and chromatography context only to their intended
  heads. The default student activation is tanh-approximated GELU; the energy encoder retains its
  small exact-GELU MLP.
- `distill/lightning.py` and `distill/context_regime.py` pass raw MS factors and dataset ids.
  Parameter-efficient context-only fitting can freeze the student backbone.
- `models/registry.py` persists `MSContextEncoder`, `ChromRunbook`, and `dataset_index` in the
  checkpoint contract; Rust export carries the corresponding tensors and metadata. Both name
  indices travel with the weights they index. The runbook owns its dataset names, and the encoder
  owns its setup names. Storing a row apart from its name would let a growing corpus renumber one while
  the other stayed put.
- Teacher batches carry their fixed acquisition factors and per-row NCE; real-data batches use
  recorded metadata, with missing energy represented as missing rather than imputed.

## Possible extensions

- **FragBackbone**: v1 is a pure projection decoder (identity backbone). Hook left to insert a
  transformer layer over fragment positions later.
- **Mobility**: single-point CCS now; distribution later.
- **Runbook sparsity**: evaluate sparse embeddings if dataset cardinality makes dense gradients
  material; the current embeddings are dense.

## Testing

- Neutral: zero/blank factors + Runbook row 0 → identical output to a context-free forward.
- Factor effect: varying NCE (through the energy MLP) moves MS2; distinct dataset rows move raw RT
  while row 0 (iRT) stays put.
- freeze_backbone: only MSContextEncoder + ChromRunbook params receive grad.
- Checkpoint round-trip: encoder + runbook + dataset_index reload and reproduce predictions.
- Parity guard: the Rust predict path (CLI `--ms-context`/`--nce`) maps to the new factors.
- Named setup: a row fitted in Rust and addressed by name predicts what the same vector predicts
  through the torch path, and an unknown name is refused rather than served from row 0.
