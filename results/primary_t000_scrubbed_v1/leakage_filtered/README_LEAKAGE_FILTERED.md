# Leakage-filtered sensitivity dataset

This output excludes matched pairs whose final judge-facing inputs contained searched presentation artifacts. It is a post-hoc sensitivity dataset, not a rerun on scrubbed prompts.

- Matched pairs seen: 250
- Clean pairs retained: 103
- Flagged pairs excluded: 147
- Primary records retained: 4120 / 10000
- Marker records retained: 4120 / 10000
- Marker decisions retained: 37080

## Pair-level flag counts

- any_presentation_artifact: 147
- llm_activity_like_fragment: 91
- llm_any_presentation_artifact: 147
- llm_avg_hr: 51
- llm_duration_text_inconsistency: 40
- llm_elevation_or_vertical: 7
- llm_ellipsis: 109
- llm_km_unit: 91
- llm_trailrun_pipe: 56

## Use in report

Use the clean-pair summaries as a robustness/sensitivity analysis: results after excluding pairs with detected presentation leakage. Do not describe these as judgments made on scrubbed prompts unless a new evaluation pass is run on scrubbed-v2 inputs.
