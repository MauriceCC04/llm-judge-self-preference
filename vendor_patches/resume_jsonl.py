"""vendor_patches/resume_jsonl.py — append-only JSONL writer with dedup by key tuple.

Upstream PR #3 to trailtraining.  Until merged, the study imports this directly.

Performance design
------------------
* The in-memory ``_seen`` set is populated ONCE (lazily on first access) by
  scanning the existing file.  All subsequent ``exists()`` / ``append()`` calls
  are O(1) against the in-memory set.
* ``_write_line`` opens the file in **append mode** rather than reading and
  rewriting the entire file.  This is O(1) per write — critical when the
  pairwise harness does 1 500+ sequential appends (250 pairs × 3 runs × 2
  positions per judge).
* Crash safety: a partially-written last line (crash mid-write) is silently
  skipped by both ``_load_seen`` and ``load_all`` via the ``json.JSONDecodeError``
  guard.  Because the ``_seen`` set is rebuilt from the file on next run, such
  a partial record will not block a retry.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

__all__ = ["ResumeJsonl", "load_all"]


class ResumeJsonl:
    """Append-only JSONL file with dedup by a caller-defined key tuple."""

    def __init__(self, path: Path, *, key_fields: Iterable[str]) -> None:
        self.path = Path(path)
        self.key_fields: tuple[str, ...] = tuple(key_fields)
        self._seen: set[tuple[Any, ...]] | None = None

    # ── Public API ──────────────────────────────────────────────────────────

    def append(self, record: dict[str, Any]) -> bool:
        """Append *record* if its key has not been seen before.

        Returns True if the record was written, False if it was a duplicate.
        """
        key = self._extract_key(record)
        seen = self._load_seen()
        if key in seen:
            return False
        self._write_line(record)
        seen.add(key)          # update in-memory set — no re-read needed
        return True

    def exists(self, record: dict[str, Any]) -> bool:
        return self._extract_key(record) in self._load_seen()

    def __len__(self) -> int:
        return len(self._load_seen())

    # ── Internal ────────────────────────────────────────────────────────────

    def _extract_key(self, record: dict[str, Any]) -> tuple[Any, ...]:
        try:
            return tuple(record[f] for f in self.key_fields)
        except KeyError as exc:
            raise KeyError(
                f"ResumeJsonl: key field {exc} missing from record. "
                f"Record keys: {list(record.keys())}"
            ) from exc

    def _load_seen(self) -> set[tuple[Any, ...]]:
        """Populate ``_seen`` from disk exactly once; return it on every call."""
        if self._seen is not None:
            return self._seen
        self._seen = set()
        if not self.path.exists():
            return self._seen
        with self.path.open(encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    self._seen.add(self._extract_key(json.loads(raw)))
                except Exception:
                    pass   # corrupt / partial line — skip silently
        return self._seen

    def _write_line(self, record: dict[str, Any]) -> None:
        """Append one JSON line to the file (O(1) — no read-rewrite)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()


def load_all(path: Path) -> list[dict[str, Any]]:
    """Load every valid JSON line from *path*; return [] if absent."""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                pass
    return out
