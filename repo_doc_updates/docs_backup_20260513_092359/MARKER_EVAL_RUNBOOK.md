# Explicit Marker-Level Evaluation Runbook

This runbook documents the secondary explicit marker-level evaluation pass over the frozen primary pairwise manifest.

## Purpose

The primary pairwise output records overall plan preference. It does not provide explicit rubric/marker scores. Earlier marker-like analyses based on keyword-coding judge rationales are exploratory only.

The explicit marker pass asks each judge to evaluate every frozen AB/BA comparison on nine markers and return structured marker-level choices and scores.

## Marker schema

Markers:

- `plan_coherence`
- `training_specificity`
- `load_progression`
- `recovery_safety`
- `quality_session_design`
- `endurance_development`
- `readiness_alignment`
- `clarity_actionability`
- `explanation_quality`

Each marker returns:

```json
{
  "preferred": "plan_a | plan_b | tie",
  "plan_a_score": 1,
  "plan_b_score": 1,
  "confidence": 0.0,
  "rationale": "brief reason"
}
```

The runner maps `plan_a` / `plan_b` back to `llm`, `programmatic`, or `tie` using manifest metadata outside the judge prompt.

## Current marker status

Current uploaded marker artifacts include:

```text
qwen_14b_judge: 2500 complete
qwen_7b_judge:  2500 complete
gemma_12b_judge: missing/pending
gemma_4b_judge:  missing/pending
```

Do not report full four-judge marker results until both Gemma marker files are complete.

## Environment

```bash
export REPO_ROOT=/mnt/beegfsstudents/home/<USER>/llm-judge-self-preference
export TRAILTRAINING_REPO=/mnt/beegfsstudents/home/<USER>/trailtraining
export PY=/home/<USER>/.conda/envs/judge-bias/bin/python
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"
```

## Runner contract

The marker SLURM wrapper must call the marker runner with `--output-dir`:

```bash
CMD=(
  "$PY" tools/run_marker_manifest_pairwise.py
  --manifest "${MANIFEST}"
  --output-dir "${OUTPUT_DIR}"
  --judge-name "${JUDGE_NAME}"
  --temperature "${JUDGE_TEMPERATURE}"
  --max-tokens "${MARKER_MAX_TOKENS:-768}"
)
```

Do not use the pairwise runner's `--output` / `--failures` arguments for marker runs.

## Launch a marker pass

Example for Gemma 12B:

```bash
MARKER_OUT="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged"
PRIMARY_MANIFEST="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl"
mkdir -p "$MARKER_OUT"

VLLM_PORT=8815 \
JUDGE_NAME=gemma_12b_judge \
MANIFEST="$PRIMARY_MANIFEST" \
OUTPUT_DIR="$MARKER_OUT" \
JUDGE_TEMPERATURE=0.0 \
TRAILTRAINING_STRUCTURED_MAX_TOKENS=768 \
MARKER_MAX_TOKENS=768 \
CLEANUP_MODEL_CACHE=0 \
sbatch slurm/run_marker_manifest_judge_hpc.sh
```

## Monitor a marker pass

Run from repo root:

```bash
watch -n 60 '
date
squeue -u "$USER"
echo
echo "rows:"
wc -l artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_gemma_12b_judge_canonical_masked_scrubbed_v1_t000.jsonl 2>/dev/null || echo missing
echo
echo "failures:"
wc -l artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/failures_marker_gemma_12b_judge_canonical_masked_scrubbed_v1_t000.jsonl 2>/dev/null || echo no-failure-file
echo
echo "last:"
tail -n 1 artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_gemma_12b_judge_canonical_masked_scrubbed_v1_t000.jsonl 2>/dev/null | jq "{record_id, judge, run, order, marker_count:(.markers|length), timestamp}" 2>/dev/null || true
'
```

If output appears missing, first confirm you are in the repo root.

## Verify a completed marker file

```bash
P="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl"

wc -l "$P"
jq -r '.record_id' "$P" | sort | uniq -d | wc -l
jq -r '.judge' "$P" | sort | uniq -c
jq -r '.order' "$P" | sort | uniq -c
jq -r '.run' "$P" | sort | uniq -c
jq -r '.markers | length' "$P" | sort | uniq -c
jq '[.markers[] | select(.normalization_warnings != null)] | length' "$P" | awk '{s+=$1} END {print "marker fields repaired:", s+0}'
```

Expected complete file:

```text
2500 rows
0 duplicate record_ids
2500 rows for one judge
1250 AB / 1250 BA
500 each run 0-4
2500 rows with 9 markers
```

## Resuming after timeout

The marker runner skips existing `record_id`s. If a job times out, simply resubmit the same judge with the same output directory.

Observed Qwen 14B marker chunks:

```text
701 -> 1404 -> 2106 -> 2500 rows
```

This is acceptable as long as final integrity checks pass.

## Missing-record manifest for final rows

If a judge file is nearly complete, build a missing-record manifest:

```bash
python - <<'PY'
import json
from pathlib import Path

judge = "qwen_7b_judge"
full = Path("artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl")
out = Path(f"artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_{judge}_canonical_masked_scrubbed_v1_t000.jsonl")
missing = Path(f"artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/missing_{judge}_marker_manifest.jsonl")

done = set()
if out.exists():
    with out.open() as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["record_id"])
rows = []
with full.open() as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("judge") == judge and r["record_id"] not in done:
            rows.append(r)
with missing.open("w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
print("done:", len(done))
print("missing:", len(rows))
print("wrote:", missing)
PY
```

Submit the missing manifest as the `MANIFEST` value.

## Preferred-field repair policy

Some Qwen 7B marker responses returned scores but blank `preferred` values. The runner repairs these deterministically:

- higher `plan_a_score` => `preferred = plan_a`
- higher `plan_b_score` => `preferred = plan_b`
- tied/missing scores => `preferred = tie`

Every repaired marker field must include `normalization_warnings`.

Current uploaded Qwen 7B marker file contains 31 repaired marker fields.

Report this in methods:

> For a small number of marker fields, the judge returned valid scores but omitted the categorical `preferred` field. These cases were repaired deterministically from the reported scores; equal or unavailable scores were coded as ties. Repaired fields were flagged with `normalization_warnings`.

## Combining all marker files

After all four marker files are complete:

```bash
D="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged"

cat "$D"/marker_gemma_4b_judge_canonical_masked_scrubbed_v1_t000.jsonl \
    "$D"/marker_gemma_12b_judge_canonical_masked_scrubbed_v1_t000.jsonl \
    "$D"/marker_qwen_7b_judge_canonical_masked_scrubbed_v1_t000.jsonl \
    "$D"/marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl \
  > "$D"/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl

wc -l "$D"/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl
jq -r '.record_id' "$D"/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl | sort | uniq -d | wc -l
jq -r '.judge' "$D"/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl | sort | uniq -c
jq -r '.markers | length' "$D"/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl | sort | uniq -c
sha256sum "$D"/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl > "$D"/marker_all_judges_canonical_masked_scrubbed_v1_t000.sha256
```

Expected final marker study:

```text
10000 marker records
0 duplicate record_ids
2500 per judge
10000 rows with 9 markers
90000 explicit marker decisions
```

## Analyze marker results

```bash
mkdir -p results/primary_t000_scrubbed_v1/marker_analysis

"$PY" tools/analyze_marker_pairwise.py \
  --marker-judgments artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl \
  --output-dir results/primary_t000_scrubbed_v1/marker_analysis
```

Priority tables:

- `marker_overall_summary.csv`
- `marker_by_judge_summary.csv`
- `marker_by_judge_family_summary.csv`
- `marker_by_source_family_summary.csv`
- `marker_by_judge_family_source_family_summary.csv`
- `marker_self_family_summary.csv`
- `marker_order_bias_summary.csv`
- `marker_pair_run_consistency_summary.csv`
- `marker_pair_run_consistency_by_judge.csv`

## Reporting rules

- Qwen-only marker results must be labeled Qwen-only.
- Full marker conclusions require all four marker files.
- Always report tie rates.
- Always report order effects and AB/BA pair-run consistency.
- Do not overclaim self-family effects unless both source and judge families are represented for that marker analysis.
