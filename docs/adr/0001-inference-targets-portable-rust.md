# Inference targets portable Rust, not Python

Prediction and spectral-library generation live in Rust. PyTorch and Lightning exist only to get
training done. Inference has to run on 32-bit musl Linux, on AWS Graviton, and possibly on wasm,
where a Python runtime is either unavailable or unwanted. Installing several GB of CUDA wheels to
execute a 500 KB model is bound by IO, not by compute. So the inference path adds no Python
dependencies, and being easier to write in Python does not justify putting anything on it.

## Consequences

- The Python `predict` command and the `msspeculator.predict` package were removed rather than
  kept in parallel with the Rust path. Row assembly, decoy generation, and library serialization
  exist once, in Rust.
- The unit that crosses from training into inference is the **portable weights** (`.safetensors`),
  not the training checkpoint. Anything inference needs must be expressible there.
- Where training needs an inference result, for diagnostics or parity checks, it calls the Rust
  implementation through the pyo3 seam instead of rewriting it in numpy.
- Torch-side prediction survives only inside training, where a live in-memory model exists and
  there is nothing to export yet.
