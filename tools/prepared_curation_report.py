"""Summarize the curation reports embedded in a prepared prefix's shard manifests.

Every shard manifest carries the policy's own report, so a finished corpus can be audited without
re-reading the spectra: per-source retention, how many shards ended up empty, how often precursor
intensity was unusable, and how often the per-run window had to be clamped into chromatographic
range. Writes JSON for the record and Markdown for review.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import fsspec


def load_manifests(prefix: str) -> list[dict[str, Any]]:
    """Read every shard manifest under a prepared prefix, read-only.

    Deliberately does not go through ``PrepareConfig``/``ensure_catalog``: this audits a corpus
    that already exists, so it must not require the current policy to match the one that built it
    (a changed policy changes the config fingerprint, which would hide every shard), and must not
    rewrite the prefix's catalog as a side effect of being asked a question about it.
    """
    fs, _, roots = fsspec.get_fs_token_paths(f"{prefix.rstrip('/')}/shards")
    try:
        paths = [path for path in fs.find(roots[0]) if path.endswith("manifest.json")]
    except FileNotFoundError:
        return []
    print(f"reading {len(paths):,} shard manifest(s) under {prefix}")

    def read(path: str) -> dict[str, Any]:
        with fs.open(path, "rb") as handle:
            return json.load(handle)

    with ThreadPoolExecutor(max_workers=16) as pool:
        return list(pool.map(read, paths))


def collect(prefix: str) -> dict[str, Any]:
    manifests = load_manifests(prefix)
    per_source: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    uncurated = 0
    empty_shards: list[str] = []

    for manifest in manifests:
        report = manifest.get("curation")
        task = manifest.get("task", {})
        source = str(task.get("dataset", "?"))
        if report is None:
            uncurated += 1
            continue
        bucket = per_source[source]
        bucket["shards"] += 1
        bucket["rows_in"] += report["input"]["rows"]
        bucket["rows_out"] += report["selection"]["selected_rows"]
        bucket["missing_intensity_rows"] += report["input"]["missing_precursor_intensity_rows"]
        bucket["peptidoforms"] += report["input"]["peptidoform_groups"]
        bucket["rejected_peptidoforms"] += report["selection"]["rejected_peptidoforms"]
        # `clamped_runs` postdates the first corpus build, so treat it as optional rather than
        # crashing on a manifest written by an older policy.
        bucket["runs"] += report["chromatography"].get("runs", 0)
        bucket["clamped_runs"] += report["chromatography"].get("clamped_runs", 0)
        if report["selection"]["selected_rows"] == 0:
            bucket["empty_shards"] += 1
            empty_shards.append(f"{task.get('source_id')}/{task.get('shard_index')}")

    # Totals come from the numeric accumulator rather than from the presentation dict below,
    # which deliberately carries None for undefined ratios and is not summable.
    def total(field: str) -> int:
        return int(sum(bucket[field] for bucket in per_source.values()))

    rows_in, rows_out = total("rows_in"), total("rows_out")
    peptidoforms = total("peptidoforms")

    per_source_out: dict[str, dict[str, int | float | None]] = {}
    for source, bucket in sorted(per_source.items()):
        source_rows = bucket["rows_in"]
        per_source_out[source] = {
            "shards": int(bucket["shards"]),
            "empty_shards": int(bucket["empty_shards"]),
            "rows_in": int(source_rows),
            "rows_out": int(bucket["rows_out"]),
            "retention": (bucket["rows_out"] / source_rows) if source_rows else None,
            "missing_intensity_fraction": (
                bucket["missing_intensity_rows"] / source_rows if source_rows else None
            ),
            "peptidoforms": int(bucket["peptidoforms"]),
            "rejected_peptidoforms": int(bucket["rejected_peptidoforms"]),
            "runs": int(bucket["runs"]),
            "clamped_runs": int(bucket["clamped_runs"]),
        }

    # Report the policy the shards were actually built with, taken from the manifests themselves.
    # A corpus should describe itself; reading it from the current config would silently relabel
    # an old corpus with today's settings.
    policies = {
        json.dumps(m["curation"]["policy"], sort_keys=True) for m in manifests if "curation" in m
    }
    return {
        "prepared_prefix": prefix,
        "policy": json.loads(next(iter(policies))) if len(policies) == 1 else None,
        "distinct_policies": len(policies),
        "shards": total("shards"),
        "manifests_without_curation": uncurated,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "retention": rows_out / rows_in if rows_in else None,
        "peptidoform_instances": peptidoforms,
        "peptidoform_instances_retained": peptidoforms - total("rejected_peptidoforms"),
        "empty_shards": empty_shards,
        "per_source": per_source_out,
    }


def render_markdown(summary: dict[str, Any], generated_on: str) -> str:
    lines = [
        "# Prepared corpus curation",
        "",
        f"**Generated by `tools/prepared_curation_report.py` on {generated_on}. Do not edit by",
        "hand — re-run the tool.**",
        "",
        f"- Prepared prefix: `{summary['prepared_prefix']}`",
        f"- Shards summarized: {summary['shards']:,}",
        "",
        "## Policy",
        "",
        "Read from the shard manifests, so this is the policy the corpus was built with rather",
        "than whatever the config says today.",
        "",
    ]
    policy = summary["policy"]
    if policy is None:
        lines.append(
            f"Shards disagree: {summary['distinct_policies']} distinct policies are present, so "
            "this prefix mixes corpora and the totals below span all of them."
        )
    else:
        lines += ["| knob | value |", "| --- | --- |"]
        lines += [f"| {key} | {value} |" for key, value in policy.items()]

    retention = summary["retention"]
    lines += [
        "",
        "## Headline",
        "",
        "Row retention is not a health metric on its own: the per-context cap deduplicates",
        "repeated observations, so a deeply-sampled source retains a smaller fraction by design.",
        "Coverage is the count of retained peptidoform instances.",
        "",
        "| | |",
        "| --- | --- |",
        f"| rows in | {summary['rows_in']:,} |",
        f"| rows retained | {summary['rows_out']:,} |",
        f"| retention | {'-' if retention is None else f'{retention:.2%}'} |",
        f"| peptidoform instances | {summary['peptidoform_instances']:,} |",
        f"| peptidoform instances retained | {summary['peptidoform_instances_retained']:,} |",
        f"| shards with zero retained rows | {len(summary['empty_shards']):,} |",
        f"| manifests without a curation report | {summary['manifests_without_curation']:,} |",
        "",
        "An empty shard is an explicit *nothing passed the policy*, which is distinguishable from",
        "a missing shard (*never processed*); finalization refuses to publish while any shard is",
        "missing.",
        "",
        "## By source",
        "",
        "| source | shards | empty | rows in | retained | no usable intensity | clamped runs |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, entry in sorted(
        summary["per_source"].items(), key=lambda kv: -(kv[1]["retention"] or 0.0)
    ):
        keep = entry["retention"]
        noint = entry["missing_intensity_fraction"]
        lines.append(
            f"| {name} | {entry['shards']:,} | {entry['empty_shards']:,} | "
            f"{entry['rows_in']:,} | {'-' if keep is None else f'{keep:.2%}'} | "
            f"{'-' if noint is None else f'{noint:.2%}'} | "
            f"{entry['clamped_runs']:,} / {entry['runs']:,} |"
        )
    if summary["empty_shards"]:
        lines += ["", "## Shards that retained nothing", ""]
        lines += [f"- `{shard}`" for shard in sorted(summary["empty_shards"])]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", help="prepared prefix, e.g. s3://.../pepdistill-prepared/v2")
    parser.add_argument("--out", type=Path, help="write the summary JSON here")
    parser.add_argument("--markdown", type=Path, help="write a committable Markdown view here")
    parser.add_argument(
        "--render-from", type=Path, help="skip reading manifests and render from a summary JSON"
    )
    args = parser.parse_args()

    if args.render_from is not None:
        summary = json.loads(args.render_from.read_text())
    elif args.prepared:
        summary = collect(args.prepared)
    else:
        raise SystemExit("--prepared is required unless --render-from is given")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(summary, date.today().isoformat()))
        print(f"wrote {args.markdown}")
    retention = summary["retention"]
    print(
        f"{summary['rows_in']:,} -> {summary['rows_out']:,} rows "
        f"({'-' if retention is None else f'{retention:.2%}'}); "
        f"{len(summary['empty_shards'])} empty shard(s)"
    )


if __name__ == "__main__":
    main()
