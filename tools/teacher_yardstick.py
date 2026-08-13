"""Measure the teacher's own agreement with our prepared real spectra, per dataset.

This is a yardstick, not a validation metric: it answers "how well does AlphaPeptDeep itself
reproduce the experimental spectra we train students against?" and therefore bounds what a
student distilled from it can be expected to reach.

It evaluates exactly the deduplicated validation winners recorded in the prepared manifest --
one spectrum per (dataset, stripped sequence, charge) -- which is the same set the student's
`val_sa/<dataset>` telemetry is computed over, so the two numbers are directly comparable.

IMPORTANT CAVEAT, recorded in the output: PROSPECT/ProteomeTools is part of AlphaPeptDeep's own
training data, and its train/validation partition is not ours. Our split is a hash of the
stripped sequence, so peptides we hold out may well be peptides the teacher trained on. Treat
these numbers as an optimistic ceiling, not a held-out measurement.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from pepdistill.data.prepared import PreparedManifest, PreparedStreamingDataset
from pepdistill.diagnostics import normalized_spectral_angle
from pepdistill.models.context import MSContextEncoder
from pepdistill.teacher import get_teacher


def _teacher_refusal(precursor) -> str | None:
    """Why the peptdeep wrapper cannot be asked about this peptide, if it cannot.

    One unaskable peptide aborts a whole batch, so classify them up front and report them as
    coverage gaps -- the gap is itself a result. The check delegates to the wrapper's own
    resolver rather than restating its rules: a second copy of "what the teacher supports"
    silently caps coverage at the stale value when the wrapper gains support.
    """
    from pepdistill.teacher.peptdeep_teacher import _alphabase_mod

    peptide = precursor.peptide
    for site, spec in peptide.mods:
        try:
            _alphabase_mod(peptide, site, spec)
        except NotImplementedError:
            return "unsupported_by_wrapper"
        except ValueError as exc:
            return (
                "mass_only_modification"
                if "mass-only" in str(exc)
                else "unresolved_modification_name"
            )
    return None


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        f"p{int(q * 100):02d}": float(np.quantile(values, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
    }


def _affine_residual_mae(observed: np.ndarray, predicted: np.ndarray) -> float | None:
    """MAE after fitting predicted ~ a*observed + b.

    The teacher's RT head has its own scale, so raw |predicted - observed| is not a meaningful
    error. Report the residual around the best affine map instead, which is scale-free.
    """
    if observed.size < 2:
        return None
    slope, intercept = np.polyfit(observed, predicted, 1)
    if not np.isfinite(slope) or slope == 0:
        return None
    return float(np.mean(np.abs(observed - (predicted - intercept) / slope)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared", required=True, help="prepared prefix, e.g. s3://.../pepdistill-prepared/v2"
    )
    parser.add_argument("--out", type=Path, default=Path("teacher-yardstick.json"))
    parser.add_argument("--teacher", default="alphapeptdeep")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--limit", type=int, default=0, help="stop after N spectra (0 = every val winner)"
    )
    args = parser.parse_args()

    manifest = PreparedManifest.load(args.prepared)
    names = {index: name for name, index in manifest.datasets.items()}
    encoder = MSContextEncoder(context_dim=8)
    dataset = PreparedStreamingDataset(manifest, encoder, frozenset({"val"}))
    teacher_kwargs = (
        {} if args.teacher == "fake" else {"device": args.device, "instrument": "Lumos"}
    )
    teacher = get_teacher(args.teacher, **teacher_kwargs)

    # peptdeep has no analyzer or activation input at all (only charge, NCE and one instrument
    # one-hot), so grouping by our own acquisition axes shows what that missing conditioning costs.
    acquisition: dict[str, list[float]] = defaultdict(list)
    angles: dict[str, list[float]] = defaultdict(list)
    rt_observed: dict[str, list[float]] = defaultdict(list)
    rt_predicted: dict[str, list[float]] = defaultdict(list)
    unsupported: dict[str, int] = defaultdict(int)
    refusals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seen = 0
    started = time.perf_counter()

    def flush(batch: list) -> None:
        nonlocal seen
        if not batch:
            return
        askable = []
        for example in batch:
            name = names.get(example.dataset_id, str(example.dataset_id))
            reason = _teacher_refusal(example.precursor)
            if reason is None:
                askable.append(example)
            else:
                unsupported[name] += 1
                refusals[name][reason] += 1
        seen += len(batch) - len(askable)
        if not askable:
            return
        precursors = [example.precursor for example in askable]
        nces = np.asarray([example.energy for example in askable], dtype=np.float32)
        labels = teacher.predict(precursors, nces=nces)
        for example, label in zip(askable, labels, strict=True):
            name = names.get(example.dataset_id, str(example.dataset_id))
            if label is None:
                unsupported[name] += 1
                refusals[name]["teacher_returned_none"] += 1
                continue
            predicted = np.asarray(label.ms2, dtype=np.float64)
            experimental = np.asarray(example.label.ms2, dtype=np.float64)
            if predicted.shape != experimental.shape:
                raise ValueError(
                    f"teacher spectrum shape {predicted.shape} != experimental "
                    f"{experimental.shape} for {example.precursor.peptide.modified_sequence()}"
                )
            angle = normalized_spectral_angle(predicted, experimental)
            angles[name].append(angle)
            cell = (
                f"{encoder.detectors[example.detector_id]}/"
                f"{encoder.fragmentations[example.fragmentation_id]}"
            )
            acquisition[cell].append(angle)
            # Label state is read from the modifications rather than the dataset name, and is
            # crossed with acquisition so a TMT penalty cannot be confused with a CID penalty.
            labelled = any(
                isinstance(spec, str) and spec.startswith("TMT")
                for _, spec in example.precursor.peptide.mods
            )
            acquisition[f"{'TMT' if labelled else 'label-free'} {cell}"].append(angle)
            if np.isfinite(label.rt) and np.isfinite(example.label.rt):
                rt_observed[name].append(float(example.label.rt))
                rt_predicted[name].append(float(label.rt))
        seen += len(askable)
        rate = seen / max(time.perf_counter() - started, 1e-9)
        print(f"  {seen:,} spectra ({rate:.0f}/s)", flush=True)

    batch: list = []
    for example in dataset.iter_examples(0, shuffle=False):
        batch.append(example)
        if len(batch) >= args.batch_size:
            flush(batch)
            batch = []
        if args.limit and seen >= args.limit:
            break
    flush(batch)

    per_dataset = {}
    for name in sorted(set(angles) | set(unsupported)):
        values = np.asarray(angles.get(name, []), dtype=np.float64)
        observed = np.asarray(rt_observed.get(name, []), dtype=np.float64)
        predicted = np.asarray(rt_predicted.get(name, []), dtype=np.float64)
        entry: dict = {
            "spectra_scored": int(values.size),
            "spectra_unsupported_by_teacher": unsupported.get(name, 0),
            "unsupported_reasons": dict(refusals.get(name, {})),
            "spectral_angle_mean": float(values.mean()) if values.size else None,
            "spectral_angle_quantiles": _quantiles(values) if values.size else None,
        }
        if observed.size >= 2:
            correlation = np.corrcoef(observed, predicted)[0, 1]
            entry["irt_r_squared"] = float(correlation**2) if np.isfinite(correlation) else None
            entry["irt_affine_residual_mae"] = _affine_residual_mae(observed, predicted)
        per_dataset[name] = entry

    every = np.concatenate([np.asarray(v) for v in angles.values()]) if angles else np.empty(0)
    report = {
        "prepared_prefix": args.prepared,
        "teacher": args.teacher,
        "metric": "normalized spectral angle, teacher prediction vs experimental spectrum",
        "evaluated_on": (
            "deduplicated validation winners from the prepared manifest "
            "(one per dataset/stripped-sequence/charge) -- the same set as student val_sa"
        ),
        "caveats": [
            "PROSPECT is part of AlphaPeptDeep's training corpus and its train/val partition is "
            "not ours, so held-out-for-us peptides may be trained-on-for-the-teacher. These are "
            "optimistic ceilings, not held-out measurements.",
            "The teacher is queried with per-row collision energy but a fixed Lumos instrument; "
            "per-row detector and fragmentation are not passed through.",
            "irt_affine_residual_mae is the residual around the best affine fit because the "
            "teacher's RT head has its own scale; raw MAE would not be comparable.",
        ],
        "val_winners_in_manifest": len(manifest.val_winners),
        "spectra_scored": int(every.size),
        "spectra_unsupported_by_teacher": int(sum(unsupported.values())),
        "unsupported_reasons": {
            reason: sum(counts.get(reason, 0) for counts in refusals.values())
            for reason in (
                "unsupported_by_wrapper",
                "mass_only_modification",
                "unresolved_modification_name",
                "teacher_returned_none",
            )
        },
        "spectral_angle_mean": float(every.mean()) if every.size else None,
        "spectral_angle_quantiles": _quantiles(every) if every.size else None,
        "per_dataset": per_dataset,
        "per_acquisition": {
            key: {
                "spectra_scored": len(values),
                "spectral_angle_mean": float(np.mean(values)),
                "spectral_angle_quantiles": _quantiles(np.asarray(values)),
            }
            for key, values in sorted(acquisition.items())
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("=" * 72)
    print(f"{'dataset':<30}{'n':>9}{'unsup':>8}{'SA':>8}{'iRT r2':>8}")
    for name, entry in sorted(
        per_dataset.items(), key=lambda kv: -(kv[1]["spectral_angle_mean"] or 0)
    ):
        sa = entry["spectral_angle_mean"]
        r2 = entry.get("irt_r_squared")
        print(
            f"{name[:29]:<30}{entry['spectra_scored']:>9,}"
            f"{entry['spectra_unsupported_by_teacher']:>8,}"
            f"{(f'{sa:.4f}' if sa is not None else '-'):>8}"
            f"{(f'{r2:.3f}' if r2 is not None else '-'):>8}"
        )
    print("=" * 72)
    print(f"{'detector/fragmentation':<30}{'n':>9}{'SA':>8}{'p50':>8}")
    for key, entry in report["per_acquisition"].items():
        print(
            f"{key:<30}{entry['spectra_scored']:>9,}"
            f"{entry['spectral_angle_mean']:>8.4f}"
            f"{entry['spectral_angle_quantiles']['p50']:>8.4f}"
        )
    print("=" * 72)
    print(f"teacher SA over {every.size:,} scored spectra: {report['spectral_angle_mean']}")
    print(f"unsupported by teacher: {report['spectra_unsupported_by_teacher']:,}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
