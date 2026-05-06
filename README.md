# LLM Judge Self-Preference Study

This repository supports a controlled empirical study of whether LLM judges systematically prefer LLM-generated training plans over programmatically generated training plans when deterministic structural plan quality is controlled.

## Research question

**Controlling for deterministic structural plan quality, do LLM judges systematically favor LLM-generated training plans over programmatically generated plans, and does this effect vary by judge model family, model size, rubric, or whether the plan matches the judge's own family?**

The estimand is a matched pairwise preference effect. Each judged item compares one LLM-source training plan against one programmatic-source training plan from the same fixture cell and with similar source-neutral structural quality.

## Current canonical study state

The current retained generation pool is larger than the earlier 512-plan baseline. The operational candidate pool now contains 1024 retained plan artifacts.

| Corpus | Path | Retained plans | Cells | Plans per cell | Notes |
|---|---:|---:|---:|---:|---|
| Qwen LLM source | `artifacts/gen_src_t070_exp_t000/full_qwen/plans` | 192 | 32 | 6 | Qwen/Qwen2.5-7B-Instruct source model |
| Gemma 3 LLM source | `artifacts/gen_src_t070_exp_t000/full_gemma3/plans` | 192 | 32 | 6 | google/gemma-3-4b-it source model |
| Programmatic control | `artifacts/gen_src_t070_exp_t000/full_programmatic/plans` | 640 | 32 | 20 | Programmatic structures plus shared explainer |
| Matching pool | `artifacts/gen_src_t070_exp_t000/matching_pool/plans` | 1024 | 32 | mixed | Combined pool for matching only |

Band balance is complete:

- Qwen: A1=48, A2=48, A3=48, A4=48.
- Gemma 3: A1=48, A2=48, A3=48, A4=48.
- Programmatic: A1=160, A2=160, A3=160, A4=160.

The current programmatic corpus passed the artifact-integrity gate:

- 640 plans and 640 provenance sidecars.
- 32 fixture cells, exactly 20 plans per cell.
- No missing sidecars.
- No bad JSON.
- No wrong plan lengths.
- No structural prompt leaks.
- No generic titles.
- No title/session mismatches.
- No rest-active contradictions.
- Plausible load gradient by athlete band.

## Important current blocker

Generation is complete. The current blocker is matching.

The first matching attempt produced only 30 matched pairs against a target of at least 250. This is not adequate for the full judge evaluation.

Diagnosis:

- Prefiltering was not the main problem. The matcher kept 936 of 1024 records and dropped only 88 session-signature duplicates.
- The old deterministic quality score produced severe source non-overlap.
- LLM plans mostly scored near 97-100.
- Programmatic plans mostly scored around 30.
- Relaxing the same-score-bin constraint did not materially increase matches.

Therefore the old TrailTraining quality score must not be used as the primary matching score unless tests prove that it is source-neutral and structural-only. It should be retained as a secondary diagnostic.

## Required methodological correction

Implement and validate a source-neutral structural matching score.

The primary matching score may use only structure-relevant fields, such as:

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
- deterministic structural violations

The primary matching score must exclude:

- title wording
- workout prose richness
- purpose prose richness
- explanation fields
- readiness rationale prose
- recovery/risk prose richness
- citations
- claim attributions
- data-note verbosity
- rationale length
- source model name
- generation arm
- file naming artifacts

A valid structural score must be invariant to prose-only edits. Tests must prove that changing title/workout/purpose/explanation text without changing structural fields does not change the structural score.

## Matching target

The target matched set is at least 250 pairs, preferably 256 pairs.

Matching should preserve causal identification:

- Match within the same fixture cell whenever possible and by default.
- Preserve same athlete band, readiness, recovery capability, race phase, style, and plan length.
- Match on source-neutral structural score and structural feature distance.
- Preserve source family metadata for analysis, but never expose source identity to judges.
- Do not blindly loosen score tolerance to inflate pair count.

Do not proceed to full judge evaluation until matching yields at least 250 valid matched pairs and the structural/style audits pass.

## 10,000 evaluation-document requirement

The full study still targets at least 10,000 judge-facing evaluation documents.

The formula is:

```text
250 matched pairs x 2 left-right orders x 5 runs x 4 judge models = 10,000 evaluation documents
```

Preferred:

```text
256 matched pairs x 2 orders x 5 runs x 4 judge models = 10,240 evaluation documents
```

The current 30-pair old-score match would produce only:

```text
30 x 2 x 5 x 4 = 1,200 evaluation documents
```

That is not sufficient and must not be described as satisfying the 10,000-document criterion.

## Current canonical next steps

1. Implement source-neutral structural scoring.
2. Add regression tests proving prose-only edits do not change structural score.
3. Rerun matching using the structural score.
4. Produce matching diagnostics and balance audits.
5. Confirm at least 250 valid matched pairs, preferably 256.
6. Run source-leakage and style-balance audits on judge-facing artifacts.
7. Generate judge-ready pair/order documents with randomized left-right order.
8. Assert the 10,000-document launch gate before full judge evaluation.
9. Run pilot judge evaluation before full 10,000-document sweep.
10. Populate the final report from actual results only.

## Repository workflow expectations

- Treat repository code and artifacts as source of truth.
- Do not fabricate results.
- Do not report generated plans as matched pairs.
- Do not report matched pairs as evaluation documents.
- Do not run full judging until the matching and launch gates pass.
- Keep all artifact-generating commands reproducible.
- Preserve provenance sidecars.
- Keep source model/family metadata for analysis, but mask it from judge-facing documents.

## Key documentation files

- `README.md`: current study state and methodological rules.
- `HPC_RUNBOOK.md`: robust HPC commands, environment setup, artifact layout, and launch gates.
- `PREREGISTRATION.md`: current preregistered estimand plus operational deviations and fixed gates.
- `INCIDENTS.md`: incident log, including current matching-score blocker.
- `TEMPERATURE_SWEEPS.md`: rules for baseline generation conditions and future temperature sweeps.
- `MATCHING_STRUCTURAL_SCORE_RUNBOOK.md`: implementation and validation checklist for the structural score.
- `EVAL_10000_LAUNCH_GATE.md`: exact 10,000-document assertion rules.
