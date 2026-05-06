# README current-study update: 1024-plan pool and 10,000-evaluation target

Status: replace or append to the existing README study summary before sending the next implementation prompt to an LLM.

## Current canonical study state

The original README described an earlier 512-plan baseline. The current retained generation pool is larger and more suitable for same-cell matching:

| Corpus | Path | Retained plans | Cells | Plans per cell |
|---|---:|---:|---:|---:|
| Qwen LLM source | `artifacts/gen_src_t070_exp_t000/full_qwen/plans` | 192 | 32 | 6 |
| Gemma 3 LLM source | `artifacts/gen_src_t070_exp_t000/full_gemma3/plans` | 192 | 32 | 6 |
| Programmatic control | `artifacts/gen_src_t070_exp_t000/full_programmatic/plans` | 640 | 32 | 20 |
| Matching pool | `artifacts/gen_src_t070_exp_t000/matching_pool/plans` | 1024 | 32 | mixed |

Band balance is complete:

- Qwen: A1=48, A2=48, A3=48, A4=48.
- Gemma 3: A1=48, A2=48, A3=48, A4=48.
- Programmatic: A1=160, A2=160, A3=160, A4=160.

The programmatic corpus has passed the current integrity gate:

- 640 plans and 640 provenance sidecars.
- 32 cells, exactly 20 plans per cell.
- No missing sidecars.
- No bad JSON.
- No wrong plan lengths.
- No structural prompt leaks.
- No generic titles.
- No title/session mismatches.
- No rest-active contradictions.

## Current blocking issue

Generation is complete. The current blocker is matching.

The first matching attempt produced only 30 matched pairs against a target of at least 250. This is not adequate for judging.

The reason is not invalid plans or prefiltering. The reason is that the existing deterministic score used for matching is not source-neutral enough for this study. It appears to reward or penalize presentation/explanation richness. LLM plans mostly score near 97-100, while programmatic plans mostly score around 30, despite programmatic plans passing structural and artifact-integrity audits.

Therefore the TrailTraining quality score must not be used as the primary matching score unless it is proven to be structural-only. Keep it as a diagnostic score.

## Required methodological correction

Implement and validate a source-neutral structural score for matching.

The primary matching score must use only structure-relevant fields, such as:

- `session_type`
- `duration_minutes`
- `is_rest_day`
- `is_hard_day`
- weekly duration/load
- long-run presence/count
- hard-day count and spacing
- rest-day count and spacing
- active-day count
- readiness/recovery/race-phase appropriateness
- deterministic structural violations

The primary matching score must exclude:

- title prose
- workout prose richness
- purpose prose richness
- explanation fields
- citations
- claim attributions
- rationale length
- model/source names
- generation arm
- file naming artifacts

## 10,000 evaluation documents requirement

The study still satisfies the intended 10,000-evaluation-document design only if matching yields at least 250 valid matched pairs.

The formula is:

```text
250 matched pairs x 2 left-right orders x 5 runs x 4 judge models = 10,000 pairwise evaluation documents
```

If matching yields 256 pairs, the evaluation volume becomes:

```text
256 matched pairs x 2 orders x 5 runs x 4 judge models = 10,240 pairwise evaluation documents
```

The current 30-pair matched set would produce only:

```text
30 x 2 x 5 x 4 = 1,200 evaluation documents
```

That is not sufficient. Do not launch full judging until the structural matching score produces at least 250 valid matched pairs, preferably 256.

## Current canonical next step

Do not generate more plans yet. Do not loosen score tolerance blindly. Do not proceed to judging.

Next implementation milestone:

1. Add a source-neutral structural scoring module.
2. Prove with tests that the score ignores presentation/prose-only changes.
3. Rerun matching using the structural score.
4. Produce matching diagnostics.
5. Proceed to style/leakage gate only if there are at least 250 valid matched pairs.
6. Launch judge jobs only after matching and style gates pass.
