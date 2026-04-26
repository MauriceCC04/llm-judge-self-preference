# HPC_SYNC_RUNBOOK.md

Canonical HPC path for the frozen 500-plan study.

This runbook supersedes the older mixed documentation because it uses the new
HPC-specific entrypoints added for this repo:

- `requirements-hpc.txt` for a pinned upstream dependency set
- `slurm/run_generation_hpc.sh` for exact-count generation
- `slurm/run_judge_hpc.sh` for robust judging with automatic style-gate refresh
- `slurm/submit_judge_hpc.sh` for judge-specific buffered walltime

## 0. What this path guarantees

Using the commands below gives you:

- a concrete audited `trailtraining` dependency pin
- exact generation totals of **250 LLM-arm + 250 programmatic-arm = 500 plans**
- automatic style-audit generation before judging when needed
- judge submission with buffered walltime derived from `judge.panel`
- no manual top-up arithmetic for the frozen study design

## 1. First-time setup on the login node

```bash
git clone <your-copy-of-this-repo>
cd llm-judge-self-preference

conda create -n judge-bias python=3.11 -y
conda activate judge-bias

# Install cluster-compatible torch / vllm first, then pinned Python deps.
pip install torch vllm
pip install -r requirements-hpc.txt

# No editable install is required for the study path.
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

Optional but recommended in your shell init on HPC:

```bash
export HF_HOME=$HOME/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PIP_CACHE_DIR=$HOME/pip_cache
export TORCH_HOME=$HOME/torch_cache
export VLLM_CACHE_ROOT=$HOME/vllm_cache
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

## 2. Gate 0 — local CPU tests

```bash
python tests/run_tests.py
```

Treat Gate 0 as mandatory before any cluster submission.

## 3. Build fixtures

```bash
python -m fixtures.build
```

## 4. Gate 1 — HPC preflight

```bash
sbatch --exclude=gnode04 --wrap='cd $PROJECT_ROOT && bash slurm/run_preflight.sh'
```

## 5. Gate 2 — vLLM smoke test

```bash
bash slurm/pre_cache_models.sh qwen_7b_judge
sbatch --exclude=gnode04 --wrap='cd $PROJECT_ROOT && bash slurm/run_vllm_smoke.sh'
```

## 6. Exact-count generation

### 6a. LLM arm — Llama source (exact 125 plans)

```bash
bash slurm/pre_cache_models.sh llama_8b_source
GENERATION_ARM=llm \
GENERATION_PROFILE=exact \
LLM_SOURCE_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    sbatch --exclude=gnode04 --wrap='cd $PROJECT_ROOT && bash slurm/run_generation_hpc.sh'
```

### 6b. LLM arm — Qwen source (exact 125 plans)

```bash
bash slurm/pre_cache_models.sh qwen_7b_source
GENERATION_ARM=llm \
GENERATION_PROFILE=exact \
LLM_SOURCE_MODEL=Qwen/Qwen2.5-7B-Instruct \
    sbatch --exclude=gnode04 --wrap='cd $PROJECT_ROOT && bash slurm/run_generation_hpc.sh'
```

After both jobs complete, the LLM arm totals **250** plans exactly.

### 6c. Fit sampler priors

```bash
python -m generate.fit_priors --plans-dir plans/ --output sampler_config.json
```

### 6d. Programmatic arm (exact 250 plans)

```bash
GENERATION_ARM=programmatic \
GENERATION_PROFILE=exact \
SAMPLER_CONFIG=sampler_config.json \
    sbatch --exclude=gnode04 --wrap='cd $PROJECT_ROOT && bash slurm/run_generation_hpc.sh'
```

At this point you should have **500 total plans**.

## 7. Matching

```bash
python cli.py match --plans plans/ --output matched_pairs.json
```

Inspect the audit output. The target remains 250 matched pairs.

## 8. Style leakage audit

You can run it explicitly once here:

```bash
python cli.py audit-style --plans plans/ --pairs matched_pairs.json --output results/
```

This is optional operationally because `slurm/run_judge_hpc.sh` will auto-build
or refresh the style audit before judging when `REQUIRE_STYLE_GATE=1`.

## 9. Pilot judge

```bash
bash slurm/pre_cache_models.sh qwen_7b_judge
JUDGE_MODE=pilot bash slurm/submit_judge_hpc.sh qwen_7b_judge
```

## 10. Full judge runs

Run one judge at a time.

```bash
bash slurm/pre_cache_models.sh llama_8b_judge
bash slurm/submit_judge_hpc.sh llama_8b_judge

bash slurm/pre_cache_models.sh qwen_7b_judge
bash slurm/submit_judge_hpc.sh qwen_7b_judge

bash slurm/pre_cache_models.sh qwen_14b_judge
bash slurm/submit_judge_hpc.sh qwen_14b_judge

bash slurm/pre_cache_models.sh qwen_32b_judge
bash slurm/submit_judge_hpc.sh qwen_32b_judge
```

The wrapper computes buffered walltime from `judge.panel.walltime_hours_with_buffer()`.

## 11. Analysis

```bash
python cli.py analyze --judgments judgments/ --plans plans/ --pairs matched_pairs.json --output results/
```

## 12. Expected frozen study totals

- LLM arm: 250 plans
- Programmatic arm: 250 plans
- Total plans: 500
- Active judges: 4
- Pairwise calls: `250 pairs × 4 judges × 5 runs × 2 positions = 10,000`
- Soft-eval calls: `500 plans × 4 judges = 2,000`

## 13. Notes on legacy files

Older files such as `slurm/run_generation.sh`, `slurm/run_judge.sh`, and the
placeholder dependency in `pyproject.toml` are not the canonical HPC path for
this frozen study. Use the files listed at the top of this runbook.
