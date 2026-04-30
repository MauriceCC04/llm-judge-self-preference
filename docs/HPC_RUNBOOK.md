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
python -m pip install --no-cache-dir torch
python -m pip install --no-cache-dir vllm
PIP_NO_CACHE_DIR=1 bash bootstrap_hpc_env.sh
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

`bash bootstrap_hpc_env.sh` is an installer, not a persistent shell initializer.
Its `export` statements do **not** modify your parent shell. Keep the cache-path
exports in your current shell before any install, cache, or `sbatch` step.

### 2e. Define quota-safe cache helpers

Under a **50 GB hard quota**, the correct rule is not “one model at a time”
literally. The correct rule is:

- keep **one required model set per job**
- remove unrelated model caches between jobs
- never accumulate the full panel in cache

In this study:

- **judge jobs** need exactly **one judge model**
- **programmatic generation jobs** need exactly the **shared explainer**
- **LLM generation jobs** need exactly the **shared explainer + one source model**

Planning budgets from `judge.panel` are:

- `Qwen/Qwen2.5-7B-Instruct`: **15 GB**
- `Qwen/Qwen2.5-3B-Instruct`: **6 GB**
- `Qwen/Qwen2.5-14B-Instruct-AWQ`: **8 GB**
- `google/gemma-3-4b-it`: **10 GB**
- `google/gemma-3-12b-it`: **28 GB**

That means:

- largest **generation** model set = `15 + 6 = 21 GB`
- largest **judge** model set = `28 GB`

Define these helpers once in your shell:

```bash
purge_cached_models() {
  python - <<'PY'
from hpc.quota import purge_all_hf_model_caches
count = purge_all_hf_model_caches()
print(f"Removed {count} cached model directories")
PY
  lquota 2>/dev/null || du -sh "${HOME}" 2>/dev/null || true
}

cache_model() {
  local model_id="$1"
  python - "$model_id" <<'PY'
from huggingface_hub import snapshot_download
import sys
model_id = sys.argv[1]
snapshot_download(model_id, ignore_patterns=["*.msgpack", "*.h5"])
print(f"Cached: {model_id}")
PY
  lquota 2>/dev/null || du -sh "${HOME}" 2>/dev/null || true
}

cache_programmatic_generation_set() {
  purge_cached_models
  cache_model "Qwen/Qwen2.5-3B-Instruct"
}

cache_llm_generation_set() {
  local source_model="$1"
  purge_cached_models
  cache_model "Qwen/Qwen2.5-3B-Instruct"
  cache_model "${source_model}"
}
```

Notes:

- `bash slurm/pre_cache_models.sh <judge_name>` is the normal helper for
  **judge-model** caching and smoke tests.
- `bash slurm/pre_cache_models.sh all` is **not** the normal quota-safe path for
  the study. It now validates/downloads models sequentially under quota and
  leaves only the **last** model cached.
- For generation jobs, do **not** call `pre_cache_models.sh` twice, because it
  purges before each download. Use the helper functions above to cache the
  explainer-plus-source set in one cycle.

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

A smoke test is a **judge-model** case, so a single-model pre-cache step is
correct here.

Do **not** use `sbatch slurm/run_vllm_smoke.sh` directly: SLURM executes a
spooled copy and the script's relative `common.sh` source will resolve under
`/var/spool/...` instead of your repo. Also do **not** use a wrapped submission
without an outer `--gres`, because the inner `#SBATCH --gres=...` will be
ignored.

```bash
cd "${REPO_ROOT}"
mkdir -p out err
bash slurm/pre_cache_models.sh qwen_7b_judge
sbatch \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
  --gres=gpu:4g.40gb:1 \
  --time=00:45:00 \
  --exclude=gnode04 \
  --chdir="${REPO_ROOT}" \
  --output=out/vllm_smoke_%j.out \
  --error=err/vllm_smoke_%j.err \
  --export=ALL,HF_HOME=${HF_HOME},HF_HUB_CACHE=${HF_HUB_CACHE},HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE},TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE},HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET} \
  --wrap="bash slurm/run_vllm_smoke.sh"
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

### 7a. LLM arm — Qwen source (exact 128 plans)

This is an **LLM generation** job, so cache the **explainer + source** model set.

```bash
cd "${REPO_ROOT}"
cache_llm_generation_set "Qwen/Qwen2.5-7B-Instruct"
mkdir -p out err
sbatch \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
  --gres=gpu:4g.40gb:1 \
  --time=16:00:00 \
  --exclude=gnode04 \
  --chdir="${REPO_ROOT}" \
  --output=out/generate_hpc_%x_%j.out \
  --error=err/generate_hpc_%x_%j.err \
  --export=ALL,HF_HOME=${HF_HOME},HF_HUB_CACHE=${HF_HUB_CACHE},HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE},TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE},HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET},GENERATION_ARM=llm,GENERATION_PROFILE=exact,LLM_SOURCE_MODEL=Qwen/Qwen2.5-7B-Instruct,SOURCE_TEMPERATURE=0.7,EXPLAINER_TEMPERATURE=0.0 \
  --wrap="bash slurm/run_generation_hpc.sh"
```

### 7b. LLM arm — Gemma source (exact 128 plans)

Before pre-caching Gemma models, make sure the Hugging Face account used on the
cluster has accepted the Gemma model terms.

This is also an **LLM generation** job, so again cache the **explainer + source**
model set.

```bash
cd "${REPO_ROOT}"
cache_llm_generation_set "google/gemma-3-4b-it"
mkdir -p out err
sbatch \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
  --gres=gpu:4g.40gb:1 \
  --time=16:00:00 \
  --exclude=gnode04 \
  --chdir="${REPO_ROOT}" \
  --output=out/generate_hpc_%x_%j.out \
  --error=err/generate_hpc_%x_%j.err \
  --export=ALL,HF_HOME=${HF_HOME},HF_HUB_CACHE=${HF_HUB_CACHE},HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE},TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE},HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET},GENERATION_ARM=llm,GENERATION_PROFILE=exact,LLM_SOURCE_MODEL=google/gemma-3-4b-it,SOURCE_TEMPERATURE=0.7,EXPLAINER_TEMPERATURE=0.0 \
  --wrap="bash slurm/run_generation_hpc.sh"
```

After both jobs complete, the LLM arm totals **256** plans exactly.

### 7c. Fit sampler priors

```bash
cd "${REPO_ROOT}"
python -m generate.fit_priors --plans-dir plans/ --output sampler_config.json
```

### 7d. Programmatic arm (exact 256 plans)

This is a **programmatic generation** job, so cache the **explainer only**.

```bash
cd "${REPO_ROOT}"
cache_programmatic_generation_set
mkdir -p out err
sbatch \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
  --gres=gpu:4g.40gb:1 \
  --time=16:00:00 \
  --exclude=gnode04 \
  --chdir="${REPO_ROOT}" \
  --output=out/generate_hpc_%x_%j.out \
  --error=err/generate_hpc_%x_%j.err \
  --export=ALL,HF_HOME=${HF_HOME},HF_HUB_CACHE=${HF_HUB_CACHE},HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE},TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE},HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET},GENERATION_ARM=programmatic,GENERATION_PROFILE=exact,SAMPLER_CONFIG=sampler_config.json,EXPLAINER_TEMPERATURE=0.0 \
  --wrap="bash slurm/run_generation_hpc.sh"
```

At this point you should have **512 total plans**.
