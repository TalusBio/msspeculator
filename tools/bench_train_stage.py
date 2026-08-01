"""Before/after benchmark for the real-data train stage.

Run against a config on this branch and on the baseline commit to compare. Reports the
numbers the streaming change is supposed to move: peak RSS and time to first batch, plus the
per-epoch cost it is allowed to add.

    uv run python tools/bench_train_stage.py runs/modrep-v2.toml --epochs 2

Requires network access to the PROSPECT S3 mirror (to extract shards on first run); a warm
FileCache makes repeat runs local. There is no offline/fake-data mode -- the point of this
harness is the real extract + meta-index + stream path.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time

from pepdistill.distill.pipeline import RunConfig, _build_train_stage, _TRAIN_SPLITS
from pepdistill.distill.real_stream import StreamingRealDataset
from pepdistill.models.context import MSContextEncoder
from pepdistill.models.registry import build_student

# ru_maxrss units are a platform fact, not a function of the value: always bytes on macOS,
# always KiB on Linux. Dispatching on the magnitude of `raw` instead (e.g. `raw > 1 << 30`) is
# wrong on macOS for any process under ~1 GB, and reports ~1000x too high. Do not "simplify"
# this back into a magnitude check.
_RSS_DIV = 1e6 if sys.platform == "darwin" else 1e3


def peak_rss_mb() -> float:
    """Peak RSS in MB. macOS reports ru_maxrss in bytes, Linux in KiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _RSS_DIV


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()

    cfg = RunConfig.from_toml(args.config)
    model = build_student(cfg.preset)
    encoder = MSContextEncoder(context_dim=model.cfg.context_dim)

    t0 = time.perf_counter()
    train_shards, _val, index, _di, _names, _stats = _build_train_stage(cfg, log=print)
    t_setup = time.perf_counter() - t0
    print(f"setup (extract + meta index)   {t_setup:8.2f}s   peak RSS {peak_rss_mb():8.1f} MB")

    ds = StreamingRealDataset(
        train_shards, index, encoder, _TRAIN_SPLITS,
        seed=cfg.seed, shuffle_buffer=cfg.train.shuffle_buffer,
    )
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        first = None
        n = 0
        for n, _ex in enumerate(ds.iter_examples(epoch), start=1):
            if first is None:
                first = time.perf_counter() - t0
        dt = time.perf_counter() - t0
        print(
            f"epoch {epoch}  first example {first:6.2f}s  total {dt:7.2f}s  "
            f"{n:,} examples  {n / dt:,.0f} ex/s  peak RSS {peak_rss_mb():8.1f} MB"
        )


if __name__ == "__main__":
    main()
