# Vendored data

`msspeculator-core` compiles these files into the binary. It does not read them at runtime. Each file
stores its provenance in its own header. This index lets you inspect the tree without opening each file.

| file | what | provenance lives in | regenerate |
| --- | --- | --- | --- |
| `elements.tsv` | 40 UNIMOD nuclide masses | `#` header | `tools/gen_unimod.py` |
| `unimod.tsv` | 1,560 UNIMOD modification deltas | `#` header | `tools/gen_unimod.py` |
| `weights/small-v0.safetensors` | trained model, embedded by `builtin.rs` | safetensors metadata + `weights/README.md` | `msspeculator export-rust` |

## The UNIMOD tables

Both are derivative works extracted from `unimod.xml`, not copies of it, redistributed under the
Design Science License. Each header records the source URL, the blake2b-256 of the exact
`unimod.xml` the rows came from, the extraction performed, the columns, and the licence.

The digest lets you check that the tables came from the stated UNIMOD source. Without it, a typo
and an upstream change would look the same as an extractor bug:

```bash
uv run python tools/gen_unimod.py --expect-digest <the source-hash in the file>
```

That fails if upstream has moved. Verified on 2026-08-25: today's `unimod.xml` (2,506,678 bytes,
blake2b-256 `30c3b621…`) reproduces both tables byte-for-byte, data rows unchanged.

A fetch date is not in the headers. The digest identifies the source. A date only records when
someone ran the script and would make every regeneration differ from the previous one.

`unimod.tsv`'s `mono_mass` column is a test fixture. The code computes mass from the composition
and checks it on every row. A bad element mass or parser bug then fails loudly instead of hiding
in an unused modification.

## Guards

`rust/core/src/unimod.rs` asserts each header still carries its source, digest, verify command,
generator and licence fields. A `#` alone would only prove that something was skipped, not that the
provenance survived a regeneration. `rust/core/src/builtin.rs` asserts the bundled weights hash to
the digest recorded beside them, and that they stay under a 4 MiB ceiling.
