"""Declarative configuration and deterministic task discovery for prepared ETL."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

#: Version of the *code* that turns a source into a shard -- the localization rule, the curation
#: window, the label columns -- as opposed to the knobs a config file can set. It participates in
#: :attr:`PrepareConfig.fingerprint`, so bumping it marks every published shard stale and the next
#: ``prepare`` run rebuilds them.
#:
#: Bump it in the same commit as any change that would make a rebuilt shard differ from a published
#: one. Without this, such a change is invisible to the staleness check: the fingerprint covers the
#: config, so a policy that moved in code leaves a corpus that reports itself complete and current
#: while holding rows the current code would never emit. Do *not* bump it for a change that cannot
#: alter shard contents -- a faster expression, an added statistic, a log line -- because every bump
#: costs a full re-prep.
#:
#: Version 1 is encoded as the absence of the field rather than as ``1``, so it reproduces the
#: fingerprints of shards published before the field existed.
#:
#: 2: rows carry one canonical ProForma column in place of a bare ``sequence`` plus a
#: ``site:spec;...`` ``mods`` string, and validation winners are deduplicated per peptidoform
#: rather than per stripped sequence.
PREPARE_POLICY_VERSION: int = 2


@dataclass(frozen=True, slots=True)
class PrepareCuration:
    """Chromatographic confidence filter applied while producing each immutable shard."""

    enabled: bool = False
    half_max_fraction: float = 0.5
    min_in_window_psms: int = 4
    max_psms_per_context: int = 2
    width_anchor_min_psms: int = 8
    energy_bucket_width: float = 1.0
    # The estimated width IS the acceptance window (apex +/- width/2), and it is estimated from
    # sampled half-height points, so it can come out implausible in both directions: a fraction of
    # a second when a run has no anchors, or minutes when a peptidoform elutes twice. Clamp it to
    # the range a real chromatographic peak can occupy.
    min_run_width_minutes: float = 0.05  # 3 s
    max_run_width_minutes: float = 0.25  # 15 s

    def __post_init__(self) -> None:
        if not 0.0 < self.half_max_fraction <= 1.0:
            raise ValueError("prepare curation half_max_fraction must be in (0, 1]")
        if self.min_in_window_psms < 1:
            raise ValueError("prepare curation min_in_window_psms must be positive")
        if self.max_psms_per_context < 1:
            raise ValueError("prepare curation max_psms_per_context must be positive")
        if self.width_anchor_min_psms < 2:
            raise ValueError("prepare curation width_anchor_min_psms must be at least two")
        if self.energy_bucket_width <= 0:
            raise ValueError("prepare curation energy_bucket_width must be positive")
        if self.min_run_width_minutes < 0:
            raise ValueError("prepare curation min_run_width_minutes must not be negative")
        if self.max_run_width_minutes < self.min_run_width_minutes:
            raise ValueError(
                "prepare curation max_run_width_minutes must not be below min_run_width_minutes"
            )

    def canonical(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PrepareSource:
    id: str
    dataset: str
    meta: str
    archive: str
    instrument: str = "Lumos"
    shards: tuple[int, ...] | str = "all"
    source_prefix: str | None = None
    record: str | None = None
    record_id: str | None = None
    archive_url: str | None = None
    meta_url: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PrepareSource":
        values = dict(raw)
        required = ("id", "dataset", "meta", "archive")
        missing = [name for name in required if name not in values]
        if missing:
            raise ValueError(f"prepare source is missing required field(s): {missing}")
        selected = values.get("shards", "all")
        if selected != "all":
            if not isinstance(selected, list) or not all(
                isinstance(value, int) and not isinstance(value, bool) for value in selected
            ):
                raise ValueError(
                    f"source {values.get('id', '<unknown>')!r}: shards must be 'all' or integers"
                )
            if len(set(selected)) != len(selected) or any(value < 0 for value in selected):
                raise ValueError(
                    f"source {values.get('id', '<unknown>')!r}: shards must be unique non-negative integers"
                )
            selected = tuple(selected)
        values["shards"] = selected
        return cls(**values)

    def canonical(self) -> dict[str, Any]:
        values = asdict(self)
        if isinstance(self.shards, tuple):
            values["shards"] = list(self.shards)
        return values


@dataclass(frozen=True, slots=True)
class PrepareGroup:
    """Archive-selection rule for one source record directory."""

    record: str
    include: tuple[str, ...] = ("*",)
    exclude: tuple[str, ...] = ()
    dataset_prefix: str | None = None
    strip_prefix: str = ""
    strip_number_suffix: bool = True
    meta_suffix: str = "_meta_data.parquet"
    instrument: str = "Lumos"
    cache_prefix: str | None = None
    source_prefix: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PrepareGroup":
        values = dict(raw)
        if "record" not in values:
            raise ValueError("prepare group is missing required field 'record'")
        for key in ("include", "exclude"):
            value = values.get(key, ("*",) if key == "include" else ())
            if isinstance(value, str):
                value = (value,)
            if not isinstance(value, (list, tuple)) or not all(
                isinstance(item, str) for item in value
            ):
                raise ValueError(f"prepare group {values['record']!r}: {key} must be a string list")
            values[key] = tuple(value)
        return cls(**values)

    def canonical(self) -> dict[str, Any]:
        values = asdict(self)
        values["include"] = list(self.include)
        values["exclude"] = list(self.exclude)
        return values


@dataclass(frozen=True, slots=True)
class PrepareConfig:
    output_prefix: str
    source_prefix: str | None = None
    cache_prefix: str | None = None
    curation: PrepareCuration = PrepareCuration()
    sources: tuple[PrepareSource, ...] = ()
    groups: tuple[PrepareGroup, ...] = ()

    @classmethod
    def load(cls, path: str | Path) -> "PrepareConfig":
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
        section = raw.get("prepare", raw)
        sources = tuple(PrepareSource.from_dict(value) for value in section.get("sources", []))
        groups = tuple(PrepareGroup.from_dict(value) for value in section.get("groups", []))
        if not sources and not groups:
            raise ValueError(
                f"{path}: [prepare] must declare [[prepare.sources]] or [[prepare.groups]]"
            )
        ids = [source.id for source in sources]
        if len(set(ids)) != len(ids):
            raise ValueError(f"prepare source ids must be unique; got {ids}")
        return cls(
            output_prefix=str(section["output_prefix"]),
            source_prefix=(str(section["source_prefix"]) if section.get("source_prefix") else None),
            cache_prefix=(
                str(section["cache_prefix"])
                if section.get("cache_prefix")
                else (str(section["source_prefix"]) if section.get("source_prefix") else None)
            ),
            curation=PrepareCuration(**section.get("curation", {})),
            sources=sources,
            groups=groups,
        )

    def canonical(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_prefix": self.source_prefix,
            "cache_prefix": self.cache_prefix,
            "output_prefix": self.output_prefix,
            "curation": self.curation.canonical(),
            "sources": [source.canonical() for source in self.sources],
            "groups": [group.canonical() for group in self.groups],
        }
        if PREPARE_POLICY_VERSION > 1:
            payload["policy_version"] = PREPARE_POLICY_VERSION
        return payload

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()
