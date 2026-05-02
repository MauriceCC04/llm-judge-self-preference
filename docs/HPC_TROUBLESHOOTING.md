# HPC_TROUBLESHOOTING.md

Failure signatures and fixes collected during real Bocconi Jupiter I setup.

Use this file when the runbook steps are followed but the cluster still behaves
in unexpected ways. Each section lists the exact error fragment that was seen,
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
python -m pip install --no-cache-dir torch
python -m pip install --no-cache-dir vllm
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

## 5. HF cache wrote to `/home` and hit quota

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

## 6. Even BeegFS home quota was not enough

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

## 7. Minimal known-good shell preamble

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
export PIP_CACHE_DIR=/mnt/beegfsstudents/home/<USER_ID>/pip_cache
export TORCH_HOME=/mnt/beegfsstudents/home/<USER_ID>/torch_cache
export VLLM_CACHE_ROOT=/mnt/beegfsstudents/home/<USER_ID>/vllm_cache
export HF_HUB_DISABLE_XET=1
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$PIP_CACHE_DIR" "$TORCH_HOME" "$VLLM_CACHE_ROOT"
cd "$REPO_ROOT"
```

## 8. `bootstrap_hpc_env.sh` completed, but later installs/downloads still used the wrong cache paths

### Failure signatures

```text
~/.cache/pip is huge
HF downloads still go to /home/<USER_ID>/...
```

### Meaning

`bash bootstrap_hpc_env.sh` ran in its own shell process. Its `export` statements
did not persist back into your interactive shell.

### Fix

Export the cache variables in your current shell before installs, downloads, or
jobs. Then clear the old pip cache:

```bash
export HF_HOME=/mnt/beegfsstudents/home/<USER_ID>/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PIP_CACHE_DIR=/mnt/beegfsstudents/home/<USER_ID>/pip_cache
export TORCH_HOME=/mnt/beegfsstudents/home/<USER_ID>/torch_cache
export VLLM_CACHE_ROOT=/mnt/beegfsstudents/home/<USER_ID>/vllm_cache
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$PIP_CACHE_DIR" "$TORCH_HOME" "$VLLM_CACHE_ROOT"
python -m pip cache purge
rm -rf ~/.cache/pip
```

## 9. The conda env did not exist yet

### Failure signatures

```text
CondaToSNonInteractiveError
Could not find conda environment: judge-bias
```

### Meaning

The environment creation failed before the env existed, often because Conda
Terms of Service had not yet been accepted.

### Fix

Accept ToS if prompted, then create the env before sourcing any shell preamble
that assumes it already exists:

```bash
module load miniconda3
eval "$(conda shell.bash hook)"
conda create -n judge-bias python=3.11 -y
conda activate judge-bias
conda info --envs
```

## 10. Gate 1 preflight failed because the explainer model was not cached

### Failure signature

```text
Model not found in cache: Qwen/Qwen2.5-3B-Instruct
```

### Meaning

Your Python environment was fine; the required explainer weights were simply not
present in the local Hugging Face cache yet.

### Fix

Cache only the explainer model on the login node, then rerun preflight:

```bash
python - <<'PY'
from hpc.quota import purge_all_hf_model_caches
count = purge_all_hf_model_caches()
print(f"Removed {count} cached model directories")
PY
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-3B-Instruct", ignore_patterns=["*.msgpack", "*.h5"])
print("Done: Qwen/Qwen2.5-3B-Instruct")
PY
```

## 11. vLLM rejected `--disable-log-requests`

### Failure signature

```text
api_server.py: error: unrecognized arguments: --disable-log-requests
```

### Meaning

Your installed vLLM expects the newer boolean flag form.

### Fix

Replace the old flag with:

```text
--no-enable-log-requests
```

in both:

- `slurm/run_vllm_smoke.sh`
- `slurm/run_generation_hpc.sh`

## 12. Wrapped smoke submission had no GPU

### Failure signature

```text
RuntimeError: No CUDA GPUs are available
```

### Meaning

You submitted the smoke script with `--wrap` but without an outer `--gres`.
The inner `#SBATCH --gres=...` inside the script was ignored.

### Fix

Submit GPU jobs with explicit outer GPU resources:

```bash
mkdir -p out err
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

## 13. Direct `sbatch slurm/run_vllm_smoke.sh` broke `common.sh`

### Failure signature

```text
/var/spool/slurmd/job.../common.sh: No such file
```

### Meaning

SLURM executed a spooled copy of the script, so `$(dirname "${BASH_SOURCE[0]}")`
resolved under `/var/spool/...` instead of your repo.

### Fix

Use the wrapped submission pattern above with `--chdir="${REPO_ROOT}"`. Do not
submit `slurm/run_vllm_smoke.sh` directly.

## 14. Smoke reached vLLM, but every compare call returned 401

### Failure signature

```text
Missing Authentication header
401
```

### Meaning

The local vLLM server was healthy, but the smoke-test client was not correctly
routed through the repo's local compatibility layer.

### Fix

In `slurm/run_vllm_smoke.sh`:

- set `TRAILTRAINING_JUDGE_LLM_BASE_URL`
- mirror it into `TRAILTRAINING_LLM_BASE_URL` and `OPENAI_BASE_URL`
- set dummy `TRAILTRAINING_JUDGE_API_KEY` and `OPENAI_API_KEY`
- call `install_trailtraining_client_compat(default_stage='judge')`

## 15. Smoke failed with `NameError: name 'judge' is not defined`

### Failure signature

```text
NameError: name 'judge' is not defined
```

### Meaning

The embedded `python -c "..."` block in the smoke script had a shell-quoting
bug around `default_stage="judge"`.

### Fix

Use single quotes inside the embedded Python:

```python
install_trailtraining_client_compat(default_stage='judge')
```

## 16. Explainer request exceeded model context length

### Failure signature

```text
This model's maximum context length is 16384 tokens. However, you requested 12288 output tokens and your prompt contains at least 4097 input tokens, for a total of at least 16385 tokens.
```

### Meaning

The explainer request budget exceeded the server's effective context limit.
This is a request-construction / server-budget problem, not evidence that the
model cannot perform the task.

### Fix

Increase the explainer server context budget and keep the validated structured
caps in place:

```bash
export TRAILTRAINING_STRUCTURED_MAX_TOKENS=12288
export TRAILTRAINING_SOURCE_MAX_TOKENS=4096
export TRAILTRAINING_EXPLAINER_MAX_TOKENS=12288
export VLLM_SOURCE_MAX_MODEL_LEN=16384
export VLLM_EXPLAINER_MAX_MODEL_LEN=24576
```

## 17. Explainer returned malformed JSON after `finish=length`

### Failure signatures

```text
finish=length
Expecting ',' delimiter
```

### Meaning

The explainer hit the output cap and was truncated. The downstream JSON parse
error is a symptom, not the root cause.

### Fix

- use the validated structured-output settings above
- prefer the working `chat.response_format + xgrammar` path
- inspect the dumped raw failure artifact before concluding that the model is
  fundamentally incapable

## 18. Saved final artifact contained `>{signal_id` placeholder leakage

### Failure signatures

```text
>{signal_id:
Placeholder leak detected in final artifact
```

### Meaning

A final saved plan artifact was contaminated by placeholder-like snapshot
content. This is study-invalid even if the JSON parses successfully.

### Fix

- build the final `snapshot` deterministically from structured signal data
- reject any saved final artifact that still contains placeholder leakage
- audit the retained corpus with a placeholder scan before matching

## 19. Manual file copying caused repo drift on HPC

### Failure signature

```text
cannot import name '_build_deterministic_snapshot'
```

### Meaning

The two repos on HPC (`llm-judge-self-preference` and `trailtraining`) were out
of sync after hand-copying some files but not others.

### Fix

- sync both repos from the desktop / GitHub copy
- clear `__pycache__`
- rerun with a fully consistent checkout

## 20. `git clean -fd` deleted generated plans and logs

### Failure signatures

```text
Removing err/
Removing out/...
Removing plans_qwen_...
```

### Meaning

Generated study artifacts were being stored as untracked files inside the repo
working tree. `git clean -fd` therefore removed them.

### Fix

- copy generated corpora to a durable non-repo artifact directory before repo cleanup
- or copy them off-cluster before any reset / clean step
- do not rely on untracked `plans_*`, `out/`, or `err/` directories inside the
  git repo as the only copy of a completed run
