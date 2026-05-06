# Structural Matching Score Runbook

## Purpose

The primary matching score must control deterministic structural quality without conditioning on source-identifying presentation style. `match/structural_score.py` implements this source-neutral score.

## Presentation-sensitivity diagnostic

Run this before treating any score as eligible for matching:

```bash
$PY tools/score_presentation_sensitivity.py \
  --plan artifacts/gen_src_t070_exp_t000/matching_pool/plans/<plan_id>.json \
  --output artifacts/gen_src_t070_exp_t000/matching_pool/presentation_sensitivity_<plan_id>.json
```

Expected for the structural score:

```text
structural_score_changed: false
```

If the legacy quality score changes after editing only title, workout prose, purpose prose, citations, claim attributions, and data notes, it is presentation-sensitive and must not be the primary matching score.

## Matching diagnostics

```bash
$PY cli.py match-diagnostics \
  --plans artifacts/gen_src_t070_exp_t000/matching_pool/plans \
  --output artifacts/gen_src_t070_exp_t000/matching_pool/structural_diagnostics
```

Inspect:

- `matching_diagnostics.json`
- `structural_scores.csv`
- `primary_tolerance_pairs.csv`
- `matched_pairs_structural_preview.json`

## Primary matching

```bash
$PY cli.py match \
  --plans artifacts/gen_src_t070_exp_t000/matching_pool/plans \
  --output artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json \
  --allow-mixed-generation-conditions \
  --tolerance 2.0 \
  --target-pairs 250 \
  --fail-below-target-ratio 1.0
```

`0.9765625` is `250 / 256`, so this command fails if the full-study minimum is not reached.

## Acceptance criteria

- matched pairs >= 250
- both Qwen and Gemma source families represented
- all 32 cells represented or missing cells justified
- mean structural score gap low
- max structural score gap <= primary tolerance
- no source labels in judge-facing payloads
- style audit passes before full judging

## Fallbacks if 250 same-cell matches are impossible

Do not run the full study. Report the failure and choose one of the following explicitly:

1. generate targeted programmatic top-up plans in under-covered structural score bands;
2. run a clearly labeled pilot on valid cells only;
3. recalibrate the structural score with a documented sensitivity analysis;
4. reduce the inferential claim to the matched subset.

Never silently relax to cross-cell matching or use presentation-sensitive scores as the primary control.
