"""Build the small source bundle consumed by ``tools/launchpad_full_run.py``.

The PROSPECT data itself is deliberately not copied here.  The cloud entrypoint reads the
already-extracted parquet objects from the S3 mirror by range request, which keeps the
Launchpad worker's 32 GB scratch volume from filling with a 461 GB dataset.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / ".launchpad" / "full-run-stage"
S3_PREFIX = "s3://terraform-workstations-bucket/jspaezp/20241022_prospect"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out: Path = args.out.resolve()

    if out.exists():
        shutil.rmtree(out)
    (out / "src").parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "src", out / "src")
    shutil.copytree(ROOT / "rust", out / "rust", ignore=shutil.ignore_patterns("target"))
    for name in ("pyproject.toml", "uv.lock", "README.md"):
        shutil.copy2(ROOT / name, out / name)

    runs = out / "runs"
    runs.mkdir()
    config = (ROOT / "runs" / "full-nontest.toml").read_text()
    config = config.replace('out = "runs/full-nontest"', 'out = "cloud-output"')
    config = config.replace('fasta = "/path/to/pretrain.fasta"', 'fasta = "pretrain.fasta"')
    config = config.replace(
        '# cache_s3_prefix = "s3://my-bucket/pepdistill-cache"',
        f'cache_s3_prefix = "{S3_PREFIX}"',
    )
    (runs / "full-nontest-cloud.toml").write_text(config)
    shutil.copy2(ROOT / "fasta_ignore" / "hela_gt20peps.fasta", out / "pretrain.fasta")

    print(f"prepared {out}")
    print("staged files: source, Rust extension, lockfile, full non-test config, pretrain FASTA")
    print(f"data source (read-only): {S3_PREFIX}")


if __name__ == "__main__":
    main()
