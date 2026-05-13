# Documentation Update Notes

This update bundle contains complete replacement markdown files for the current study state. It is intended for use when the repository docs are stale relative to the completed primary pairwise run and the partial explicit marker-level run.

## Files to update

| File | Reason to update |
|---|---|
| `README.md` | Current state, primary results, partial marker status, reproducibility warnings |
| `HPC_RUNBOOK.md` | Correct HPC paths, direct env Python, model cache/quota workflow, primary and marker commands |
| `PREREGISTRATION.md` | Frozen design, analysis plan, explicit marker extension, current result-disclosure boundaries |
| `INCIDENTS.md` | Matching-score failure, environment errors, vLLM flag issue, cache/quota issues, marker runner bugs, repair policy |
| `TEMPERATURE_SWEEPS.md` | Clarifies primary `t000` status and separation of future sensitivity runs |
| `MARKER_EVAL_RUNBOOK.md` | New explicit marker-level procedure and analysis guidance |
| `RESULTS_SUMMARY.md` | Primary results and current Qwen-only explicit marker results |
| `ARTIFACT_MANIFEST.md` | Frozen artifacts, checksums, hygiene items, and current marker file status |

## Files intentionally not updated

No other docs were updated because the uploaded frozen artifact bundle did not contain the checked-in documentation tree, and the GitHub connector was not enabled. Avoid editing unknown docs blindly.

## How to apply

Copy this directory to your repository root or run the script from wherever the markdown replacements are located:

```bash
bash apply_doc_updates.sh
```

Then inspect:

```bash
git diff -- README.md HPC_RUNBOOK.md PREREGISTRATION.md INCIDENTS.md TEMPERATURE_SWEEPS.md MARKER_EVAL_RUNBOOK.md RESULTS_SUMMARY.md ARTIFACT_MANIFEST.md
```

## Current factual state encoded in these docs

- Primary 10,000 pairwise judgments are complete.
- Primary overall LLM win rate is 36.53%.
- Programmatic plans win 63.47% row-level overall.
- Position effects are large: AB 57.54% LLM win rate vs BA 15.52%.
- Pair-run source-consistent programmatic wins exceed source-consistent LLM wins: 1,973 vs 626.
- Explicit marker-level results are complete for Qwen 7B and Qwen 14B only in the uploaded artifact bundle.
- Current marker archive has 5,000 marker records / 45,000 marker decisions, not the full 10,000 / 90,000.
- Qwen-only marker results favor programmatic plans on every explicit marker.
- Qwen-only marker results do not show robust Qwen self-family preference.
- Gemma marker runs remain pending/missing from the uploaded marker archive.
