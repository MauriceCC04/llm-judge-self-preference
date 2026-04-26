# HPC_SYNC_RUNBOOK.md

Canonical HPC path for the frozen 512-plan study.

This runbook supersedes the older mixed documentation because it uses the code-first
entrypoints for the frozen study.

## 0. What this path guarantees

Using the commands below gives you:

- a concrete audited `trailtraining` dependency pin
- exact generation totals of **256 LLM-arm + 256 programmatic-arm = 512 plans**
- automatic style-audit generation before judging when needed
- judge submission with buffered walltime derived from `judge.panel`
- optional `PAIRWISE_VIEW=canonical_masked` control runs

## 1. First-time setup on the login node

```bash
git clone <your-copy-of-this-repo>
cd llm-judge-self-preference

conda create -n judge-bias python=3.11 -y
conda activate judge-bias

pip install torch vllm
bash bootstrap_hpc_env.sh
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

## 2. Gate 0 — local CPU tests

```bash
python tests/run_tests.py
```

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

### 6a. LLM arm — Llama source (exact 128 plans)

```bash
bash slurm/pre_cache_models.sh llama_8b_source
GENERATION_ARM=llm GENERATION_PROFILE=exact LLM_SOURCE_MODEL=meta-llama/Llama-3.1-8B-Instruct     sbatch --exclude=gnode04 --wrap='cd $PROJECT_ROOT && bash slurm/run_generation_hpc.sh'
```

### 6b. LLM arm — Qwen source (exact 128 plans)

```bash
bash slurm/pre_cache_models.sh qwen_7b_source
GENERATION_ARM=llm GENERATION_PROFILE=exact LLM_SOURCE_MODEL=Qwen/Qwen2.5-7B-Instruct     sbatch --exclude=gnode04 --wrap='cd $PROJECT_ROOT && bash slurm/run_generation_hpc.sh'
```

After both jobs complete, the LLM arm totals **256** plans exactly.

### 6c. Fit sampler priors

```bash
python -m generate.fit_priors --plans-dir plans/ --output sampler_config.json
```

### 6d. Programmatic arm (exact 256 plans)

```bash
GENERATION_ARM=programmatic GENERATION_PROFILE=exact SAMPLER_CONFIG=sampler_config.json     sbatch --exclude=gnode04 --wrap='cd $PROJECT_ROOT && bash slurm/run_generation_hpc.sh'
```

At this point you should have **512 total plans**.

## 7. Matching

```bash
python cli.py match --plans plans/ --output matched_pairs.json
```

Inspect the audit output. The target remains 250 matched pairs.

## 8. Style leakage audit

```bash
python cli.py audit-style --plans plans/ --pairs matched_pairs.json --output results/
```

## 9. Pilot judge

```bash
bash slurm/pre_cache_models.sh qwen_7b_judge
JUDGE_MODE=pilot bash slurm/submit_judge_hpc.sh qwen_7b_judge
```

## 10. Full judge runs

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

To run the stricter masked control view:

```bash
PAIRWISE_VIEW=canonical_masked bash slurm/submit_judge_hpc.sh qwen_7b_judge
```

## 11. Analysis

```bash
python cli.py analyze --judgments judgments/ --plans plans/ --pairs matched_pairs.json --output results/
```

## 12. Expected frozen study totals

- LLM arm: 256 plans
- Programmatic arm: 256 plans
- Total plans: 512
- Active judges: 4
- Pairwise calls: `250 pairs × 4 judges × 5 runs × 2 positions = 10,000`
- Soft-eval calls: `512 plans × 4 judges = 2,048`
