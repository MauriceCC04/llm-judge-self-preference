# Structural Matching Score Runbook

This runbook defines the required correction to the matching pipeline.

## 1. Problem

The old matcher used an existing TrailTraining quality score as the primary deterministic matching score. On the 1024-plan matching pool, it produced only 30 matched pairs.

Diagnostics showed:

- Qwen plans mostly scored near 97-100.
- Gemma 3 plans mostly scored near 97-100.
- Programmatic plans mostly scored around 30.
- Programmatic plans passed artifact-integrity and structural audits.
- Relaxing same-bin matching did not solve the problem.

Conclusion: the old quality score is likely contaminated by presentation/explanation richness and is not valid as the primary matching score for this study.

## 2. Required score properties

The primary matching score must be source-neutral and structural.

It may use:

- `session_type`
- `duration_minutes`
- `is_rest_day`
- `is_hard_day`
- weekly duration/load
- active-day count
- rest-day count and spacing
- hard-day count and spacing
- long-run presence/count
- quality-session presence/count
- readiness/recovery/race-phase appropriateness
- structural violations

It must exclude:

- title prose
- workout prose richness
- purpose prose richness
- explanation text
- readiness rationale prose
- recovery/risk prose richness
- citations
- claim attributions
- data-note verbosity
- source labels
- source model names
- file names

## 3. Suggested implementation

Add a module such as:

```text
match/structural_score.py
```

Suggested API:

```python
@dataclass
class StructuralScoreReport:
    score: float
    version: str
    components: dict[str, float]
    features: dict[str, Any]
    violations: list[str]


def score_structural_plan(
    plan_path: Path,
    provenance_path: Path | None = None,
    rollups_path: Path | None = None,
) -> StructuralScoreReport:
    ...
```

The exact implementation should use existing repository utilities where appropriate.

## 4. Component suggestions

A defensible 0-100 structural score can combine:

- basic validity and seven-day structure
- duration/load appropriateness by athlete band
- active/rest-day balance
- long-run appropriateness
- hard-day/quality-session appropriateness
- rest and hard-day spacing
- race-phase appropriateness
- readiness/recovery appropriateness
- contradiction and impossible-duration penalties

The score does not need to imitate TrailTraining's prose-sensitive quality score. It must control deterministic structural quality for causal identification.

## 5. Matching update

Update the matcher to use structural score as primary:

- write `structural_score` to provenance or matching records
- write `structural_score_version`
- retain old quality score only as `diagnostic_quality_score` if useful
- compute `score_gap` from structural scores
- match within same fixture cell by default
- use weighted structural feature distance as a tie-breaker

Do not use generation arm, source family, title/prose length, or file names as matching features.

## 6. Required tests

Add tests proving:

1. The structural score is unchanged when only title/workout/purpose prose changes.
2. The structural score changes when duration/session/rest/hard fields change.
3. LLM and programmatic plans can be scored by the same function.
4. The report includes version, components, features, and violations.
5. Matching on synthetic structurally-overlapping data produces expected pair counts.
6. Judge-facing pair artifacts contain no source labels.
7. Left-right randomization is reproducible with a seed.

## 7. Diagnostics after implementation

Run and save:

- structural score distribution by arm/source model
- per-fixture structural score overlap
- possible pair counts by tolerance
- final matched-pair count
- final coverage by fixture and band
- Qwen/Gemma source representation
- structural feature balance table
- score gap mean/max/p95
- style/leakage audit

## 8. Success criteria

Full judging may proceed only if:

```text
matched_pairs >= 250
preferably matched_pairs >= 256
structural score gap is low and bounded
coverage across all 32 cells is documented
Qwen and Gemma are represented
style/leakage gate passes
```

If this cannot be achieved, do not proceed to full judge evaluation. Instead report the bottleneck and choose a defensible fallback.
