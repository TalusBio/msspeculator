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
from datetime import date
from pathlib import Path
from typing import Any

import fsspec
import numpy as np

from pepdistill.data.prepared import load_shard_manifests
from pepdistill.diagnostics import SA_HISTOGRAM_BINS, SA_HISTOGRAM_EDGES, SpectralAngleSeries


def collect(prefix: str) -> dict[str, Any]:
    manifests = load_shard_manifests(prefix, log=print)
    per_source: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    uncurated = 0
    empty_shards: list[str] = []
    # Per-dataset achievable ceiling. Histogram counts over a shared grid are additive, so shards
    # sum to the exact corpus distribution -- no averaging of per-shard means, which would weight
    # a shard with ten replicate comparisons the same as one with ten thousand.
    ceilings: dict[str, dict[str, np.ndarray]] = defaultdict(
        lambda: defaultdict(lambda: np.zeros(SA_HISTOGRAM_BINS, dtype=np.int64))
    )

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
        # Absent from corpora built before the ceiling became mandatory.
        ceiling = report.get("achievable_ceiling") or {}
        for subset in ("all", "within_apex_window", "selected"):
            histogram = (ceiling.get(subset) or {}).get("histogram")
            if histogram:
                ceilings[source][subset] += np.asarray(histogram["counts"], dtype=np.int64)

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
        for subset, counts in ceilings.get(source, {}).items():
            series = SpectralAngleSeries(subset, [int(count) for count in counts])
            per_source_out[source][f"ceiling_{subset}_mean"] = series.mean()
            per_source_out[source][f"ceiling_{subset}_replicates"] = series.total()

    # Report the policy the shards were actually built with, taken from the manifests themselves.
    # A corpus should describe itself; reading it from the current config would silently relabel
    # an old corpus with today's settings.
    policies = {
        json.dumps(m["curation"]["policy"], sort_keys=True) for m in manifests if "curation" in m
    }
    # The knobs above are only half the policy; the rest is the code that applied them, which the
    # config cannot express. Shards built before the version was recorded report it as absent
    # rather than as a number, since guessing one would claim knowledge the manifest does not have.
    versions = sorted(
        {m.get("task", {}).get("policy_version", "unrecorded") for m in manifests}, key=str
    )
    return {
        "prepared_prefix": prefix,
        "policy": json.loads(next(iter(policies))) if len(policies) == 1 else None,
        "policy_versions": versions,
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
        # Kept separate from per_source so the tables above stay scalar and this stays directly
        # usable as a violin series: same grid as the published teacher yardstick and the
        # student's validation histograms.
        "achievable_ceiling": {
            "metric": (
                "leave-one-out spectral angle of each replicate against the consensus of the "
                "other replicates in its acquisition context"
            ),
            "histogram_bin_edges": list(SA_HISTOGRAM_EDGES),
            "per_source": {
                source: {
                    subset: [int(count) for count in counts]
                    for subset, counts in sorted(subsets.items())
                }
                for source, subsets in sorted(ceilings.items())
            },
        },
    }


def render_markdown(summary: dict[str, Any], generated_on: str) -> str:
    lines = [
        "# Prepared corpus curation",
        "",
        f"**Generated by `tools/prepared_curation_report.py` on {generated_on}. Do not edit by",
        "hand — re-run the tool.**",
        "",
        "- Corpus: prepared-v2 snapshot (internal storage location omitted)",
        "- Reproduction record: the generated JSON retains the exact prepared prefix",
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
        versions = ", ".join(str(version) for version in summary["policy_versions"])
        lines.append(f"| policy_version (code) | {versions} |")

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
        "| source | shards | empty shards | rows in | rows retained (%) |"
        " rows without usable intensity (%) | runs clamped / runs |"
        " ceiling SA mean (in-window) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, entry in sorted(
        summary["per_source"].items(), key=lambda kv: -(kv[1]["retention"] or 0.0)
    ):
        keep = entry["retention"]
        noint = entry["missing_intensity_fraction"]
        # The in-window subset, matching the series the training panel draws: the retained subset
        # is capped at two PSMs, so its leave-one-out score is pairwise and understates the bound.
        ceiling = entry.get("ceiling_within_apex_window_mean")
        lines.append(
            f"| {name} | {entry['shards']:,} | {entry['empty_shards']:,} | "
            f"{entry['rows_in']:,} | {'-' if keep is None else f'{keep:.2%}'} | "
            f"{'-' if noint is None else f'{noint:.2%}'} | "
            f"{entry['clamped_runs']:,} / {entry['runs']:,} | "
            f"{'-' if ceiling is None else f'{ceiling:.4f}'} |"
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
    parser.add_argument(
        "--publish",
        action="store_true",
        help="also write the summary to <prepared>/diagnostics/curation-summary.json",
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
    if args.publish:
        # Beside the teacher yardstick, so a training run finds both reference lines in one place.
        destination = (
            f"{str(summary['prepared_prefix']).rstrip('/')}/diagnostics/curation-summary.json"
        )
        with fsspec.open(destination, "w") as handle:
            handle.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"published {destination}")
    retention = summary["retention"]
    print(
        f"{summary['rows_in']:,} -> {summary['rows_out']:,} rows "
        f"({'-' if retention is None else f'{retention:.2%}'}); "
        f"{len(summary['empty_shards'])} empty shard(s)"
    )


if __name__ == "__main__":
    main()
