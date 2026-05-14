# Incidents, Fixes, and Final Provenance Notes

This document records the major operational and methodological incidents encountered during the completed Qwen/Gemma study, along with their fixes and final status.

## Incident 1: Original matcher produced only 30 pairs

### Symptom

The original matching run produced only about 30 matched pairs from the 1,024-plan matching pool, far below the 250-pair gate for the full study.

### Root cause

The deterministic score used by the matcher was not suitable as a source-neutral structural score. Programmatic plans mostly scored around 30, while LLM-source plans mostly scored 97-100. Relaxing tolerance did not solve the issue.

### Fix

A source-neutral structural score was implemented and used for matching. The final frozen matched set contains 250 matched pairs with low structural gaps.

### Final status

Resolved. The full 10,000-record study uses the frozen 250-pair structural matched set.

## Incident 2: Environment errors on HPC

### Symptoms

Observed errors included:

```text
ModuleNotFoundError: No module named 'pydantic'
ModuleNotFoundError: No module named 'trailtraining'
broken conda activate
wrong Python interpreter
```

### Fix

Use explicit environment variables and direct Python path:

```bash
export REPO_ROOT=/mnt/beegfsstudents/home/3202029/llm-judge-self-preference
export TRAILTRAINING_REPO=/mnt/beegfsstudents/home/3202029/trailtraining
export PY=/home/3202029/.conda/envs/judge-bias/bin/python
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"
```

### Final status

Resolved.

## Incident 3: Model cache and quota constraints

### Symptom

Quota limitations prevented all models from being cached simultaneously. Gemma gated models also required Hugging Face authentication.

### Fix

Models were cached and run one at a time. Gated Gemma models required `HF_TOKEN` and accepted model terms.

### Final status

Resolved. All four primary and marker judge outputs are complete.

## Incident 4: vLLM flag mismatch for Qwen 14B AWQ

### Symptom

Qwen 14B failed with a quantization mismatch:

```text
Quantization method specified in the model config (awq) does not match the quantization argument (awq_int4)
```

### Fix

The vLLM quantization argument was normalized to `awq` for `Qwen/Qwen2.5-14B-Instruct-AWQ`.

### Final status

Resolved.

## Incident 5: Qwen 14B structured-output token cap

### Symptom

Qwen 14B failed because the runner requested 4096 output tokens with a 4096-token model context.

### Fix

Set:

```bash
TRAILTRAINING_STRUCTURED_MAX_TOKENS=768
MARKER_MAX_TOKENS=768
```

### Final status

Resolved.

## Incident 6: Long Qwen 14B runtimes and partial chunks

### Symptom

Qwen 14B marker runs timed out before reaching 2,500 rows.

### Fix

The marker runner is append/resume-safe and skips existing `record_id`s. Qwen 14B was completed over multiple chunks.

### Final status

Resolved. Final Qwen 14B marker file has 2,500 rows, 0 duplicate record IDs, and 9 markers per row.

## Incident 7: Marker SLURM output path bug

### Symptom

The marker runner wrote outputs into a nested directory named like a pairwise `.jsonl` file.

### Root cause

The copied SLURM script still used old pairwise-style `--output` and `--failures` arguments instead of the marker runner's `--output-dir` argument.

### Fix

The marker SLURM `CMD` block was patched to call:

```bash
"$PY" tools/run_marker_manifest_pairwise.py \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --judge-name "${JUDGE_NAME}" \
  --temperature "${JUDGE_TEMPERATURE}" \
  --max-tokens "${MARKER_MAX_TOKENS:-768}"
```

### Final status

Resolved. Final marker outputs are in the intended directory.

## Incident 8: Qwen 7B blank `preferred` fields in marker output

### Symptom

A small number of Qwen 7B marker responses left `preferred` blank for a marker.

### Fix

The marker runner was patched to repair blank or invalid `preferred` values from `plan_a_score` and `plan_b_score`. If scores were equal or unavailable, the marker was coded as `tie`. Repaired marker fields were flagged with `normalization_warnings`.

### Final status

Resolved. Final dataset has 31 repaired marker fields out of 90,000 marker decisions.

## Incident 9: Gemma malformed JSON in marker retries

### Symptom

Gemma 12B and Gemma 4B first-pass marker runs produced a small number of JSON decode failures.

### Fix

The marker runner was patched to request compact JSON, use `response_format={"type":"json_object"}`, and set rationale fields to empty strings for retry runs. Failed rows were rerun from missing-record manifests.

### Final status

Resolved. Final Gemma 12B and Gemma 4B marker files each have 2,500 rows and no current failures.

## Incident 10: Source-asymmetric presentation leakage

### Symptom

A post hoc leakage audit found that source masking removed explicit metadata but did not fully remove presentation-level artifacts. LLM-source plans sometimes included:

```text
TrailRun |
avgHR
km
m+ / elevation / vertical
ellipses
duration-text inconsistencies
```

These artifacts were absent from programmatic plans in the scan.

### Fix

No scrubbed-v2 rerun was performed due to time constraints. Instead, an exclusion-based leakage-filtered sensitivity dataset was created by excluding any matched pair with these artifacts.

### Final status

Partially mitigated analytically. This remains a limitation and must be reported clearly.

Final leakage-filtered sensitivity state:

- 103 clean pairs retained
- 147 flagged pairs excluded
- 4,120 clean primary records
- 37,080 clean marker decisions
- Main conclusions unchanged in the clean subset

## Final incident summary

All blocking operational incidents were resolved. The only remaining methodological caveat is presentation leakage, which is now documented and handled through an exclusion-based sensitivity analysis. The study should be treated as complete for the current frozen Qwen/Gemma design.
