# HPC_RUNBOOK.md — Judge-Bias Study on Bocconi Jupiter I

---

## 0. Cluster facts

| Item | Value |
|---|---|
| Cluster | Bocconi Jupiter I |
| Partition / QoS | `stud` |
| GPU | one MIG A100 slice `gpu:4g.40gb:1` (40 GB VRAM) |
| Home quota | 50 GB — **50 GB hard ceiling, 30 GB soft ceiling in scripts** |
| Job submission limit | 2 running, 2 submitted (QoS policy) |
| Known bad node | `gnode04` — always `--exclude=gnode04` on every `sbatch` |

---

## 1. First-time setup (login node only)

```bash
# 1. Clone and install
git clone https://github.com/<your-fork>/judge-bias-study.git
cd judge-bias-study
conda create -n judge-bias python=3.11 -y
conda activate judge-bias

pip install -e /path/to/trailtraining          # local editable install
pip install vllm torch                          # pre-built HPC wheel if available
pip install -e ".[dev,analysis]"                # study extras: numpy, pandas,
                                                # statsmodels, scipy, matplotlib

# 2. Set cache paths (also hardcoded in slurm/common.sh — must match)
export HF_HOME=$HOME/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PIP_CACHE_DIR=$HOME/pip_cache
export TORCH_HOME=$HOME/torch_cache
export VLLM_CACHE_ROOT=$HOME/vllm_cache
export HF_HUB_DISABLE_TELEMETRY=1

# 3. Pre-register (commit PREREGISTRATION.md BEFORE any generation sbatch)
git add PREREGISTRATION.md && git commit -m "pre-registration: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 4. Build fixtures (idempotent — already committed, safe to re-run)
python -m fixtures.build
```

---

## 2. Gate sequence

### Gate 0 — Local tests (~2 min on laptop, no GPU needed)

```bash
python tests/run_tests.py
```

**All 33 tests must pass before any `sbatch` run.** This catches import errors,
schema drift, logic bugs, and pipeline breaks using CPU-only mock LLM calls.

### Gate 1 — HPC preflight (~5 min on compute node)

```bash
sbatch --exclude=gnode04 \
    --wrap='cd $PROJECT_ROOT && bash slurm/run_preflight.sh'
```

Checks trailtraining imports, GPU visibility, storage headroom, and one
mock vLLM call.

### Gate 2 — vLLM smoke test (~30 min)

```bash
# Pre-cache Qwen2.5-7B from login node first (only CDN access is on login node)
bash slurm/pre_cache_models.sh qwen_7b_judge

sbatch --exclude=gnode04 \
    --wrap='cd $PROJECT_ROOT && bash slurm/run_vllm_smoke.sh'
```

Starts vLLM with Qwen2.5-7B, calls `compare_plans` on one pair, verifies JSON
output shape, shuts down.

---

## 3. Generation

### 3a. LLM arm — first batch (~100 plans, used to fit sampler priors)

```bash
bash slurm/pre_cache_models.sh llama_8b_source
GENERATION_ARM=llm GENERATION_N=7 sbatch --exclude=gnode04 \
    --wrap='cd $PROJECT_ROOT && bash slurm/run_generation.sh'
# Wait for completion, then:
bash slurm/pre_cache_models.sh qwen_7b_source
GENERATION_ARM=llm GENERATION_N=7 sbatch --exclude=gnode04 \
    --wrap='cd $PROJECT_ROOT && bash slurm/run_generation.sh'
```

This produces ~7 plans/fixture × 2 source models × 8 fixtures = ~112 plans.

### 3b. Fit sampler priors

```bash
python cli.py fit-priors --plans plans/ --output sampler_config.json
# or equivalently:
python -m generate.fit_priors --plans-dir plans/ --output sampler_config.json
```

Inspect the output — `p_hard_day` should be 0.2–0.4, `p_rest_day` 0.10–0.20.
If the priors look pathological, generate more LLM plans before proceeding.

### 3c. LLM arm — remaining plans

```bash
# Run with seed_offset=7 so new plans don't overwrite existing ones
GENERATION_ARM=llm GENERATION_N=13 sbatch --exclude=gnode04 \
    --wrap='cd $PROJECT_ROOT && SEED_OFFSET=7 bash slurm/run_generation.sh'
```

### 3d. Programmatic arm

```bash
GENERATION_ARM=programmatic SAMPLER_CONFIG=sampler_config.json \
    GENERATION_N=40 sbatch --exclude=gnode04 \
    --wrap='cd $PROJECT_ROOT && bash slurm/run_generation.sh'
```

---

## 4. Matching

```bash
python cli.py match --plans plans/ --output matched_pairs.json
```

Inspect the audit printout:
- Pairs yielded ≥ 200 (target 250)?  If not, re-tune sampler priors and regenerate.
- Mean score gap ≤ 0.5?

---

## 5. Gate 3 — Pilot judge (30 pairs, 1 judge)

```bash
bash slurm/pre_cache_models.sh qwen_7b_judge

# Pilot: 30 pairs only (-—pilot flag)
JUDGE_NAME=qwen_7b_judge sbatch --exclude=gnode04 \
    --wrap='cd $PROJECT_ROOT && bash slurm/run_judge.sh --pilot'

# Check pilot bias gate manually after completion:
python -c "
import sys; sys.path.insert(0, '.')
from judge.harness import check_pilot_bias_gate
from pathlib import Path
g = check_pilot_bias_gate(Path('judgments/pairwise_qwen_7b_judge.jsonl'))
print(g['message'])
"
```

**If the gate fails** (|P(prefer_a) − 0.5| ≥ 0.2), log in INCIDENTS.md and
exclude this judge from H1/H2 analysis.  Proceed with remaining judges.

---

## 6. Gate 4 — Full judge runs (5 judges × 6–16 h each)

Run one judge at a time to stay within the 2-job submission limit.  After
each judge completes, delete its model weights before caching the next.

```bash
for judge in llama_8b_judge qwen_14b_judge qwen_32b_judge mistral_7b_judge; do
    bash slurm/pre_cache_models.sh $judge
    JUDGE_NAME=$judge sbatch --exclude=gnode04 \
        --wrap="cd \$PROJECT_ROOT && bash slurm/run_judge.sh"
    echo "Submitted $judge — wait for completion before next"
done
```

After all five judges complete:
```bash
lquota          # should be back near pre-judge usage
ls -lh judgments/
wc -l judgments/*.jsonl
```

---

## 7. Analysis

```bash
python cli.py analyze \
    --judgments judgments/ \
    --plans plans/ \
    --pairs matched_pairs.json \
    --output results/

# or equivalently:
python -m analyze.run_analysis \
    --judgments judgments/ --plans plans/ --output results/
```

Output:
```
results/
├── summary.json           ← H1–H4 verdicts + all model outputs
├── summary.md             ← human-readable table with ✅/❌ per hypothesis
├── h1_logistic_result.json
├── h1_forest.png
├── h2_rubric_deltas.json
├── h2_rubric_deltas.csv
├── h2_rubric_heatmap.png
├── h3_self_preference.json
├── h4_scale_result.json
├── h4_scale_curve.png
├── position_bias_audit.csv
├── position_bias_audit.png
├── schema_failure_rates.csv
└── pair_coverage.csv
```

---

## 8. Disk discipline

| Phase | Expected peak usage | Action if exceeded |
|---|---|---|
| Before generation | ~2 GB | — |
| During LLM generation | ~2 GB + one model (15 GB) | Delete model after job |
| After generation | ~4 GB | — |
| During judging | ~4 GB + one judge (8–18 GB) | Delete model after job |
| After all judging | ~5 GB | — |

**Always check `lquota` before and after every generation/judge job.**

If a job crashes mid-download and leaves orphaned weights:
```bash
bash tools/clean_caches.sh
```

---

## 9. Resume after crash

All JSONL writers (`PairwiseWriter`, `SoftEvalWriter`) are crash-safe:
dedup keys prevent re-writing completed records.  Simply re-submit the
failed job:

```bash
JUDGE_NAME=qwen_14b_judge sbatch --exclude=gnode04 \
    --wrap='cd $PROJECT_ROOT && bash slurm/run_judge.sh'
```

The harness will skip already-written records and continue from the
first missing (pair × run × position) triple.

Generation is similarly resumable: `run_generation.py` skips any
`plan_id.json` that already exists in the output directory.

---

## 10. Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| vLLM OOM | `max-model-len` too large | Reduce `--max-model-len` in `run_judge.sh` |
| `gnode04` job hangs | Bad node | Always `--exclude=gnode04` |
| quota exceeded | Orphaned HF download | `bash tools/clean_caches.sh` |
| Pilot bias gate FAIL | Judge has strong position bias | Exclude from H1/H2; log INC |
| `<50%` pair coverage | Sampler priors far from LLM arm | Re-fit priors with more plans |
| `statsmodels` missing | `[analysis]` extra not installed | `pip install -e ".[analysis]"` |
