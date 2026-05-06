"""Diagnostic: does the legacy quality score change under presentation-only edits?"""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any

from match.pair import _old_quality_score
from match.structural_score import score_structural_plan, score_structural_plan_obj


def mutate_presentation_only(plan: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(plan)
    out["citations"] = ["presentation-only diagnostic citation"] * 5
    out["claim_attributions"] = [{"claim": "presentation diagnostic", "support": "presentation diagnostic"}]
    out["data_notes"] = ["presentation-only data note with many words"] * 8
    meta = out.setdefault("meta", {})
    meta["primary_goal"] = "Presentation-only diagnostic title change that must not affect structure"
    for idx, day in enumerate((out.get("plan") or {}).get("days") or []):
        if not isinstance(day, dict):
            continue
        day["title"] = f"Presentation diagnostic title {idx} with extra wording"
        day["purpose"] = "This is deliberately verbose presentation text. " * 8
        day["workout"] = "This is deliberately verbose workout prose. " * 10
        day.setdefault("signal_ids", [])
    return out


def run(path: Path, *, rollups_path: Path | None = None) -> dict[str, Any]:
    original = json.loads(path.read_text(encoding="utf-8"))
    mutated = mutate_presentation_only(original)
    prov_path = path.with_name(path.name + ".provenance.json")
    provenance = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.exists() else None

    structural_original = score_structural_plan(path, provenance_path=prov_path if prov_path.exists() else None).score
    structural_mutated = score_structural_plan_obj(mutated, provenance, plan_id=path.stem).score

    old_original = None
    old_mutated = None
    old_error = None
    with tempfile.TemporaryDirectory() as td:
        mutated_path = Path(td) / path.name
        mutated_path.write_text(json.dumps(mutated, ensure_ascii=False), encoding="utf-8")
        try:
            old_original = _old_quality_score(path, rollups_path, strict=True)
            old_mutated = _old_quality_score(mutated_path, rollups_path, strict=True)
        except Exception as exc:
            old_error = f"{type(exc).__name__}: {exc}"

    return {
        "plan_path": str(path),
        "structural_score_original": structural_original,
        "structural_score_mutated": structural_mutated,
        "structural_score_changed": structural_original != structural_mutated,
        "legacy_quality_score_original": old_original,
        "legacy_quality_score_mutated": old_mutated,
        "legacy_quality_score_changed": (old_original is not None and old_mutated is not None and old_original != old_mutated),
        "legacy_quality_score_error": old_error,
        "interpretation": (
            "Legacy score changed under presentation-only mutation; do not use it for primary structural matching."
            if old_original is not None and old_mutated is not None and old_original != old_mutated
            else "Legacy score did not change on this diagnostic case or could not be run; inspect additional plans before treating it as structural-only."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--rollups", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    result = run(Path(args.plan), rollups_path=Path(args.rollups) if args.rollups else None)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    if result["structural_score_changed"]:
        raise SystemExit("Structural score changed under presentation-only mutation; this is a bug.")


if __name__ == "__main__":
    main()
