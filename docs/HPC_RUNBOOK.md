# HPC_RUNBOOK.md

Canonical HPC path for the frozen 512-plan baseline study.

This runbook supersedes older mixed documentation because it uses the code-first
entrypoints for the frozen study. It is intentionally **baseline-first**.

For temperature sensitivity runs, see `docs/TEMPERATURE_SWEEPS.md`.
For concrete failure signatures and fixes collected during real Bocconi setup,
see `docs/HPC_TROUBLESHOOTING.md`.

## 0. What this path guarantees

Using the commands below gives you:

- a concrete audited `trailtraining` dependency pin
- exact generation totals of **256 LLM-arm + 256 programmatic-arm = 512 plans**
- automatic style-audit generation before judging when needed
- judge submission with buffered walltime derived from `judge.panel`
- optional `PAIRWISE_VIEW=canonical_masked` control runs
- baseline temperature settings:
  - source generation temperature `0.7`
  - shared explainer temperature `0.0`
  - judge temperature `0.0`

This runbook describes the **real HPC path**. It does **not** use the local
`tools/mock_llm_server.py` smoke path.

## 1. HPC assumptions that matter for reproducibility

This repository is reproducible on Bocconi Jupiter I only if the following are
kept stable across runs:

- the repo checkout is current enough to pass Gate-0
- the Python interpreter is the intended conda environment, not the system
  Miniconda interpreter and not `~/.local` user-site packages
- CUDA is loaded before installing `vllm`
- Hugging Face caches are stored on a quota-appropriate filesystem rather than
  the small `/home/<USER_ID>` default
- generation and judging are launched through the SLURM wrappers, not by calling
  `cli.py generate` or `cli.py judge` directly on the login node without
  endpoint wiring

If any of those assumptions are violated, you can end up with apparently subtle
failures that are actually environment mistakes. See
`docs/HPC_TROUBLESHOOTING.md` for exact error strings.

## 2. First-time setup on the login node

Choose a real repo path on the cluster and keep using it consistently.
On Bocconi, wrapped `sbatch` commands should `cd` to the real repo path rather
than relying on the job spool directory.

### 2a. Clone and define the repo root

```bash
git clone <your-copy-of-this-repo> /mnt/beegfsstudents/home/<USER_ID>/llm-judge-self-preference
export REPO_ROOT=/mnt/beegfsstudents/home/<USER_ID>/llm-judge-self-preference
cd "${REPO_ROOT}"
```

### 2b. Create and activate the Python environment correctly

Do **not** use plain `conda activate` on Bocconi login shells unless the conda
shell hook has already been loaded. The reproducible pattern is:

```bash
module load miniconda3
eval "$(conda shell.bash hook)"
conda create -n judge-bias python=3.11 -y
conda activate judge-bias
export PYTHONNOUSERSITE=1
```

Verify immediately:

```bash
which python
python --version
python -c "import sys; print(sys.executable)"
```

Expected interpreter shape:

```text
~/.conda/envs/judge-bias/bin/python
Python 3.11.x
```

If you instead see `/software/miniconda3/...`, `cp313` wheels, or imports from
`~/.local/lib/python3.13/...`, you are not in the intended environment.

### 2c. Load CUDA before installing `vllm`

`vllm` installation on Jupiter I should be done only after loading CUDA and
exporting `CUDA_HOME`.

```bash
module load cuda/12.4
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
which nvcc
echo "$CUDA_HOME"
```

Then install the key runtime packages in the active env:

```bash
python -m pip install torch
python -m pip install vllm
bash bootstrap_hpc_env.sh
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
```

### 2d. Move Hugging Face caches off `/home`

On Bocconi, the default Hugging Face cache location under `/home/<USER_ID>` is
not suitable for gated 8B and judge-model downloads. Set the cache location
explicitly before pre-caching any model.

```bash
export HF_HOME=/mnt/beegfsstudents/home/<USER_ID>/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
```

Check quota before downloading large models:

```bash
lquota
```

If the quota is too tight for multiple models, plan for a **one-model-at-a-time
workflow** or move caches to a larger project/scratch area if available.

## 3. Gate 0 — local CPU tests

```bash
cd "${REPO_ROOT}"
python tests/run_tests.py
```

This gate must be green before moving on.

If Gate-0 reports wrapper-script parse failures such as:

- `run_generation.sh does not source common.sh`
- `run_judge.sh does not source common.sh`
- `submit_judge.sh does not source common.sh`

then your checkout is stale or locally patched in a way that is inconsistent
with the current runbook. Sync the repo to a passing revision before
continuing.

## 4. Build fixtures

```bash
cd "${REPO_ROOT}"
python -m fixtures.build
```

## 5. Gate 1 — HPC preflight

```bash
sbatch \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
  --exclude=gnode04 \
  --wrap="cd ${REPO_ROOT} && bash slurm/run_preflight.sh"
```

## 6. Gate 2 — vLLM smoke test

```bash
cd "${REPO_ROOT}"
bash slurm/pre_cache_models.sh qwen_7b_judge
sbatch \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
  --exclude=gnode04 \
  --export=ALL,HF_HOME=${HF_HOME},HF_HUB_CACHE=${HF_HUB_CACHE},HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE},TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE},HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET} \
  --wrap="cd ${REPO_ROOT} && bash slurm/run_vllm_smoke.sh"
```

## 7. Exact-count generation

### Important: do not launch direct CLI generation on HPC without endpoint wiring

For the real HPC path, prefer the SLURM wrappers. Calling `python cli.py generate`
directly on the login node is only valid if you have already started the local
vLLM endpoints yourself and exported the stage-specific environment variables:

- `TRAILTRAINING_SOURCE_LLM_BASE_URL`
- `TRAILTRAINING_EXPLAINER_LLM_BASE_URL`
- `TRAILTRAINING_JUDGE_LLM_BASE_URL`

If those are missing or stale, you can hit failures such as:

- `Missing Authentication header`
- `No explainer endpoint configured`

Use the wrapper path below instead.

### 7a. LLM arm — Llama source (exact 128 plans)

Before pre-caching gated Llama models, make sure the Hugging Face account used
on the cluster has actually been approved for the model repo.

```bash
cd "${REPO_ROOT}"
hf auth login
bash slurm/pre_cache_models.sh llama_8b_source
sbatch \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
  --exclude=gnode04 \
  --export=ALL,HF_HOME=${HF_HOME},HF_HUB_CACHE=${HF_HUB_CACHE},HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE},TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE},HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET},GENERATION_ARM=llm,GENERATION_PROFILE=exact,LLM_SOURCE_MODEL=meta-llama/Llama-3.1-8B-Instruct,SOURCE_TEMPERATURE=0.7,EXPLAINER_TEMPERATURE=0.0 \
  --wrap="cd ${REPO_ROOT} && bash slurm/run_generation_hpc.sh"
```

### 7b. LLM arm — Qwen source (exact 128 plans)

```bash
cd "${REPO_ROOT}"
bash slurm/pre_cache_models.sh qwen_7b_source
sbatch \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
  --exclude=gnode04 \
  --export=ALL,HF_HOME=${HF_HOME},HF_HUB_CACHE=${HF_HUB_CACHE},HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE},TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE},HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET},GENERATION_ARM=llm,GENERATION_PROFILE=exact,LLM_SOURCE_MODEL=Qwen/Qwen2.5-7B-Instruct,SOURCE_TEMPERATURE=0.7,EXPLAINER_TEMPERATURE=0.0 \
  --wrap="cd ${REPO_ROOT} && bash slurm/run_generation_hpc.sh"
```

After both jobs complete, the LLM arm totals **256** plans exactly.

### 7c. Fit sampler priors

```bash
cd "${REPO_ROOT}"
python -m generate.fit_priors --plans-dir plans/ --output sampler_config.json
```

### 7d. Programmatic arm (exact 256 plans)

```bash
cd "${REPO_ROOT}"
sbatch \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
  --exclude=gnode04 \
  --export=ALL,HF_HOME=${HF_HOME},HF_HUB_CACHE=${HF_HUB_CACHE},HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE},TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE},HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET},GENERATION_ARM=programmatic,GENERATION_PROFILE=exact,SAMPLER_CONFIG=sampler_config.json,EXPLAINER_TEMPERATURE=0.0 \
  --wrap="cd ${REPO_ROOT} && bash slurm/run_generation_hpc.sh"
```

At this point you should have **512 total plans**.

## 8. Matching

```bash
cd "${REPO_ROOT}"
python cli.py match --plans plans/ --output matched_pairs.json
```

Inspect the audit output. The target remains 250 matched pairs.

## 9. Style leakage audit

```bash
cd "${REPO_ROOT}"
python cli.py audit-style --plans plans/ --pairs matched_pairs.json --output results/
```

## 10. Pilot judge

Start with a real pilot before launching the full panel.
This is where you validate endpoint compatibility, startup stability, provenance,
and output shape against a real vLLM server.

```bash
cd "${REPO_ROOT}"
bash slurm/pre_cache_models.sh qwen_7b_judge
JUDGE_MODE=pilot JUDGE_TEMPERATURE=0.0 bash slurm/submit_judge_hpc.sh qwen_7b_judge
```

Recommended pilot checks after the job completes:

- nonempty `judgments/` outputs
- no schema-failure growth
- no provenance exclusions during `cli.py analyze`
- successful pairwise and soft-eval record loading

## 11. Dedicated `qwen_32b_judge` validation before the full panel

Before launching the full four-judge study, validate the 32B AWQ judge on your
actual `gpu:4g.40gb:1` slice with a small pilot shard.

```bash
cd "${REPO_ROOT}"
bash slurm/pre_cache_models.sh qwen_32b_judge
PAIR_LIMIT=5 PLAN_LIMIT=10 RUN_SOFT_EVAL=1 RUN_PAIRWISE=1 JUDGE_MODE=pilot JUDGE_TEMPERATURE=0.0 \
  bash slurm/submit_judge_hpc.sh qwen_32b_judge
```

If this fails for memory or startup reasons, adjust the judge launch settings
before starting the full four-judge panel.

## 12. Full judge runs

```bash
cd "${REPO_ROOT}"

bash slurm/pre_cache_models.sh llama_8b_judge
JUDGE_TEMPERATURE=0.0 bash slurm/submit_judge_hpc.sh llama_8b_judge

bash slurm/pre_cache_models.sh qwen_7b_judge
JUDGE_TEMPERATURE=0.0 bash slurm/submit_judge_hpc.sh qwen_7b_judge

bash slurm/pre_cache_models.sh qwen_14b_judge
JUDGE_TEMPERATURE=0.0 bash slurm/submit_judge_hpc.sh qwen_14b_judge

bash slurm/pre_cache_models.sh qwen_32b_judge
JUDGE_TEMPERATURE=0.0 bash slurm/submit_judge_hpc.sh qwen_32b_judge
```

To run the stricter masked control view:

```bash
cd "${REPO_ROOT}"
PAIRWISE_VIEW=canonical_masked JUDGE_TEMPERATURE=0.0 bash slurm/submit_judge_hpc.sh qwen_7b_judge
```

## 13. Analysis

```bash
cd "${REPO_ROOT}"
python cli.py analyze --judgments judgments/ --plans plans/ --pairs matched_pairs.json --output results/
```

## 14. Expected frozen study totals

- LLM arm: 256 plans
- Programmatic arm: 256 plans
- Total plans: 512
- Active judges: 4
- Pairwise calls: `250 pairs × 4 judges × 5 runs × 2 positions = 10,000`
- Soft-eval calls: `512 plans × 4 judges = 2,048`

## 15. Notes on local smoke vs HPC

A local smoke run with `tools/mock_llm_server.py` is useful for validating client
wiring and end-to-end plumbing on a laptop, but it is not part of the HPC path.

Key differences:

- local smoke uses a synthetic HTTP mock server
- HPC uses real vLLM servers launched by the SLURM scripts
- local mock runs can show empty H1/H2 after position-bias exclusion
- HPC pilots should be treated as the real compatibility gate before the full run

## 16. Quota and model-cache strategy

Model caching is the most likely storage bottleneck on Bocconi.

Recommended order of operations:

1. pre-cache exactly the model you need next
2. run the jobs that depend on it
3. delete that cache if quota becomes binding
4. pre-cache the next model

If quota is exceeded, remove the partial model cache and its lock directory
before retrying. Exact examples are recorded in `docs/HPC_TROUBLESHOOTING.md`.

## 17. Temperature sensitivity runs

The baseline runbook above is the canonical frozen-study path.

For temperature sensitivity analyses:

- do **not** mix multiple generation conditions in one `plans/` directory
- do **not** run matching on a mixed-temperature directory
- do use a separate artifact root per generation condition
- do sweep judge temperature on fixed matched plans
- see `docs/TEMPERATURE_SWEEPS.md` for the recommended workflow
