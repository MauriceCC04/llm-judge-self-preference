# HPC_TROUBLESHOOTING.md

Failure signatures and fixes collected during real Bocconi Jupiter I setup.

Use this file when the runbook steps are followed but the cluster still behaves
in unexpected ways.  Each section lists the exact error fragment that was seen,
what it actually meant, and the least-surprising fix.

## 1. Conda activation did not really activate the env

### Failure signature

```text
CondaError: Run 'conda init' before 'conda activate'
```

Later signs of the same problem:

```text
cp313 wheels
~/.local/lib/python3.13/...
/software/miniconda3/...
```

### Meaning

You are not actually inside the intended `judge-bias` environment, even if the
shell prompt looks plausible.

### Fix

```bash
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate judge-bias
export PYTHONNOUSERSITE=1
which python
python --version
```

Expected interpreter:

```text
~/.conda/envs/judge-bias/bin/python
Python 3.11.x
```

## 2. `vllm` install failed with missing CUDA

### Failure signature

```text
AssertionError: CUDA_HOME is not set
```

### Meaning

`vllm` was being installed before the CUDA module was loaded.

### Fix

```bash
module load cuda/12.4
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
which nvcc
echo "$CUDA_HOME"
python -m pip install torch
python -m pip install vllm
```

## 3. Gate-0 failed because wrapper scripts did not source `common.sh`

### Failure signatures

```text
run_generation.sh does not source common.sh
run_judge.sh does not source common.sh
submit_judge.sh does not source common.sh
```

### Meaning

Your checkout is stale or locally inconsistent with the current runbook.

### Fix

Sync the repository to a revision that passes Gate-0, then rerun:

```bash
python tests/run_tests.py
```

Do not continue to HPC jobs until Gate-0 is green.

## 4. Direct `cli.py generate` failed on HPC

### Failure signatures

```text
Missing Authentication header
No explainer endpoint configured
```

### Meaning

You called `python cli.py generate ...` directly without a live local-vLLM
endpoint and without the stage-specific environment variables that the SLURM
wrappers normally set.

### Fix

Use the canonical wrapper path instead:

```bash
bash slurm/pre_cache_models.sh <model_alias>
sbatch ... --wrap="cd ${REPO_ROOT} && bash slurm/run_generation_hpc.sh"
```

Only use direct CLI generation if you have already started the local vLLM server
and exported the appropriate stage URL variables:

- `TRAILTRAINING_SOURCE_LLM_BASE_URL`
- `TRAILTRAINING_EXPLAINER_LLM_BASE_URL`
- `TRAILTRAINING_JUDGE_LLM_BASE_URL`

## 5. Hugging Face login worked, but gated Llama access still failed

### Failure signatures

```text
401 Unauthorized
not in the authorized list
```

### Meaning

Your token is valid, but the HF account used on the cluster does not yet have
approval for the gated model repository.

### Fix

Request access on the model page, wait for approval, then relogin:

```bash
hf auth login
bash slurm/pre_cache_models.sh llama_8b_source
```

## 6. Xet-backed HF downloads failed mid-cache

### Failure signature

```text
File reconstruction error: Internal Writer Error: Background writer channel closed
```

### Meaning

The `hf-xet` path was unstable for this download on this cluster.

### Fix

Disable Xet before the cache step:

```bash
export HF_HUB_DISABLE_XET=1
bash slurm/pre_cache_models.sh llama_8b_source
```

Optional fallback if needed:

```bash
export HF_HUB_DISABLE_XET=1
export HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1
bash slurm/pre_cache_models.sh llama_8b_source
```

## 7. HF cache wrote to `/home` and hit quota

### Failure signatures

```text
Ignored error while writing commit hash to /home/<USER_ID>/hf_cache/...
OSError: [Errno 122] Disk quota exceeded: '/home/<USER_ID>/hf_cache/hub/.locks/...'
```

### Meaning

The default Hugging Face cache location under `/home/<USER_ID>` is too small for
these models.

### Fix

Move the cache first, then retry:

```bash
export HF_HOME=/mnt/beegfsstudents/home/<USER_ID>/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
rm -rf /home/<USER_ID>/hf_cache
```

## 8. Even BeegFS home quota was not enough

### Failure signatures

```text
USED=51.5GB LIMIT=50GB
26G /mnt/beegfsstudents/home/<USER_ID>/hf_cache
```

### Meaning

The cache relocation worked, but the total study footprint still exceeded the
available 50 GB BeegFS home quota.

### Immediate recovery

Stop any jobs still writing, then remove the partial model cache and locks:

```bash
squeue -u <USER_ID>
scancel <jobid>
rm -rf /mnt/beegfsstudents/home/<USER_ID>/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct
rm -rf /mnt/beegfsstudents/home/<USER_ID>/hf_cache/hub/.locks/models--meta-llama--Llama-3.1-8B-Instruct
lquota
```

### Long-term fix

One of the following is required:

- a larger scratch/project cache location than BeegFS home
- or a one-model-at-a-time workflow:
  1. pre-cache one model
  2. run the jobs that need it
  3. delete that cache
  4. pre-cache the next model

## 9. Minimal known-good shell preamble

Use this before HPC cache or run steps:

```bash
module load miniconda3 cuda/12.4
eval "$(conda shell.bash hook)"
conda activate judge-bias
export PYTHONNOUSERSITE=1
export REPO_ROOT=/mnt/beegfsstudents/home/<USER_ID>/llm-judge-self-preference
export HF_HOME=/mnt/beegfsstudents/home/<USER_ID>/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_DISABLE_XET=1
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
cd "$REPO_ROOT"
```
