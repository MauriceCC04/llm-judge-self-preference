"""judge/outputs.py — JSONL writers with dedup discipline."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Literal
from vendor_patches.resume_jsonl import ResumeJsonl

Position = Literal["AB", "BA"]


class PairwiseWriter:
    KEY_FIELDS = ("pair_id", "judge", "run", "position")
    def __init__(self, path: Path) -> None:
        self._store = ResumeJsonl(path, key_fields=self.KEY_FIELDS)
    def append(self, record: dict[str, Any]) -> bool:
        return self._store.append(record)
    def exists(self, record: dict[str, Any]) -> bool:
        return self._store.exists(record)
    def __len__(self) -> int:
        return len(self._store)


class SoftEvalWriter:
    KEY_FIELDS = ("plan_id", "judge")
    def __init__(self, path: Path) -> None:
        self._store = ResumeJsonl(path, key_fields=self.KEY_FIELDS)
    def append(self, record: dict[str, Any]) -> bool:
        return self._store.append(record)
    def exists(self, record: dict[str, Any]) -> bool:
        return self._store.exists(record)
    def __len__(self) -> int:
        return len(self._store)


class SchemaFailWriter:
    KEY_FIELDS = ("plan_id", "judge", "call_type")
    def __init__(self, path: Path) -> None:
        self._store = ResumeJsonl(path, key_fields=self.KEY_FIELDS)
    def append(self, record: dict[str, Any]) -> bool:
        return self._store.append(record)
    def __len__(self) -> int:
        return len(self._store)
