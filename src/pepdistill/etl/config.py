"""Declarative configuration and deterministic task discovery for prepared ETL."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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
            sources=sources,
            groups=groups,
        )

    def canonical(self) -> dict[str, Any]:
        return {
            "source_prefix": self.source_prefix,
            "cache_prefix": self.cache_prefix,
            "output_prefix": self.output_prefix,
            "sources": [source.canonical() for source in self.sources],
            "groups": [group.canonical() for group in self.groups],
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()
