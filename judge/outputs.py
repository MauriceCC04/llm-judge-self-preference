"""judge/outputs.py — JSONL writers with dedup discipline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from vendor_patches.resume_jsonl import ResumeJsonl

Position = Literal["AB", "BA"]


def _normalize_judge_temperature(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    raw = normalized.get("judge_temperature", 0.0)
    try:
        normalized["judge_temperature"] = float(raw)
    except Exception:
        normalized["judge_temperature"] = 0.0
    return normalized


class PairwiseWriter:
    KEY_FIELDS = ("pair_id", "judge", "judge_temperature", "run", "position", "pairwise_view")

    def __init__(self, path: Path) -> None:
        self._store = ResumeJsonl(path, key_fields=self.KEY_FIELDS)

    def _normalize(self, record: dict[str, Any]) -> dict[str, Any]:
        normalized = _normalize_judge_temperature(record)
        normalized.setdefault("pairwise_view", "raw_normalized")
        return normalized

    def append(self, record: dict[str, Any]) -> bool:
        return self._store.append(self._normalize(record))

    def exists(self, record: dict[str, Any]) -> bool:
        return self._store.exists(self._normalize(record))

    def __len__(self) -> int:
        return len(self._store)


class SoftEvalWriter:
    KEY_FIELDS = ("plan_id", "judge", "judge_temperature")

    def __init__(self, path: Path) -> None:
        self._store = ResumeJsonl(path, key_fields=self.KEY_FIELDS)

    def _normalize(self, record: dict[str, Any]) -> dict[str, Any]:
        return _normalize_judge_temperature(record)

    def append(self, record: dict[str, Any]) -> bool:
        return self._store.append(self._normalize(record))

    def exists(self, record: dict[str, Any]) -> bool:
        return self._store.exists(self._normalize(record))

    def __len__(self) -> int:
        return len(self._store)


class SchemaFailWriter:
    KEY_FIELDS = ("record_id", "judge", "judge_temperature", "call_type", "run", "position", "pairwise_view")

    def __init__(self, path: Path) -> None:
        self._store = ResumeJsonl(path, key_fields=self.KEY_FIELDS)

    def _normalize(self, record: dict[str, Any]) -> dict[str, Any]:
        normalized = _normalize_judge_temperature(record)
        normalized.setdefault("record_id", normalized.get("pair_id") or normalized.get("plan_id") or "")
        normalized.setdefault("run", -1)
        normalized.setdefault("position", "")
        normalized.setdefault("pairwise_view", "")
        return normalized

    def append(self, record: dict[str, Any]) -> bool:
        return self._store.append(self._normalize(record))

    def exists(self, record: dict[str, Any]) -> bool:
        return self._store.exists(self._normalize(record))

    def __len__(self) -> int:
        return len(self._store)