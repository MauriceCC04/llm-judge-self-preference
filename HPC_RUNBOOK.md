
# HPC_SYNC_RUNBOOK.md

Canonical Bocconi HPC path for the frozen 512-plan study.

This runbook is the single source of truth for syncing, validating, running,
monitoring, and recovering the judge-bias study on Bocconi Jupiter I.

It also reflects the operational lessons from Bocconi HPC:

- use absolute repo-path wrapped `sbatch` commands
- prefer sequential launch discipline on `stud`
- pre-cache model weights on the login node
- keep home usage comfortably below the hard quota before long runs

---

## Cluster assumptions

- Cluster: Bocconi Jupiter I
- Partition / QoS: `stud` / `stud`
- GPU request: `gpu:4g.40gb:1`
- Account: `3202029`
- Home quota: 50 GB total
- Recommended soft ceiling before long runs: about 40 GB

---

## 0. What this path guarantees

Using the commands below gives you:

- a concrete audited `trailtraining` dependency pin
- exact generation totals of **256 LLM-arm + 256 programmatic-arm = 512 plans**
- automatic style-audit generation before judging when needed
- judge submission with buffered walltime derived from `judge.panel`
- optional `PAIRWISE_VIEW=canonical_masked` control runs
- one-at-a-time sequential judge launching on `stud`

---

## 1. First-time setup on the login node

```bash
ssh <USER>@slogin.hpc.unibocconi.it
cd /mnt/beegfsstudents/home/<USER>/llm-judge-self-preference

module load miniconda3
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n judge-bias python=3.11 -y
conda activate judge-bias

pip install torch vllm
bash bootstrap_hpc_env.sh

export REPO_ROOT=$PWD
export PYTHONPATH=$PWD:${PYTHONPATH:-}
````

---

## 2. Gate 0 — local CPU tests

```bash
python tests/run_tests.py
```

---

## 3. Build fixtures

```bash
python -m fixtures.build
```

---

## 4. Gate 1 — HPC preflight

Always use a wrapped `sbatch` with an absolute repo path.

```bash
sbatch \
  --account=3202029 \
  --partition=stud \
  --qos=stud \
  --exclude=gnode04 \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --wrap="cd ${REPO_ROOT} && bash slurm/run_preflight.sh"
```

---

## 5. Gate 2 — vLLM smoke test

```bash
bash slurm/pre_cache_models.sh qwen_7b_judge

sbatch \
  --account=3202029 \
  --partition=stud \
  --qos=stud \
  --exclude=gnode04 \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --wrap="cd ${REPO_ROOT} && bash slurm/run_vllm_smoke.sh"
```

---

## 6. Exact-count generation

Run one generation job at a time.

### 6a. LLM arm — Llama source (exact 128 plans)

```bash
bash slurm/pre_cache_models.sh llama_8b_source

GENERATION_ARM=llm \
GENERATION_PROFILE=exact \
LLM_SOURCE_MODEL=meta-llama/Llama-3.1-8B-Instruct \
sbatch \
  --account=3202029 \
  --partition=stud \
  --qos=stud \
  --exclude=gnode04 \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --wrap="cd ${REPO_ROOT} && bash slurm/run_generation_hpc.sh"
```

### 6b. LLM arm — Qwen source (exact 128 plans)

```bash
bash slurm/pre_cache_models.sh qwen_7b_source

GENERATION_ARM=llm \
GENERATION_PROFILE=exact \
LLM_SOURCE_MODEL=Qwen/Qwen2.5-7B-Instruct \
sbatch \
  --account=3202029 \
  --partition=stud \
  --qos=stud \
  --exclude=gnode04 \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --wrap="cd ${REPO_ROOT} && bash slurm/run_generation_hpc.sh"
```

After both jobs complete, the LLM arm totals **256** plans exactly.

### 6c. Fit sampler priors

```bash
python -m generate.fit_priors --plans-dir plans/ --output sampler_config.json
```

### 6d. Programmatic arm (exact 256 plans)

```bash
GENERATION_ARM=programmatic \
GENERATION_PROFILE=exact \
SAMPLER_CONFIG=sampler_config.json \
sbatch \
  --account=3202029 \
  --partition=stud \
  --qos=stud \
  --exclude=gnode04 \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --wrap="cd ${REPO_ROOT} && bash slurm/run_generation_hpc.sh"
```

At this point you should have **512 total plans**.

---

## 7. Matching

```bash
python cli.py match --plans plans/ --output matched_pairs.json
```

Inspect the audit output. The target remains 250 matched pairs.

---

## 8. Style leakage audit

```bash
python cli.py audit-style --plans plans/ --pairs matched_pairs.json --output results/
```

---

## 9. Pilot judge

Recommended first pilot:

```bash
bash slurm/pre_cache_models.sh qwen_7b_judge
WAIT_FOR_JOB=1 JUDGE_MODE=pilot bash slurm/submit_judge_hpc.sh qwen_7b_judge
```

If the pilot fails the bias gate, do not proceed to full runs for that judge.

---

## 10. Full judge runs

### Recommended path: sequential panel launcher

This launcher:

* runs on the login node
* pre-caches one judge at a time
* submits one judge job
* waits for completion
* moves to the next judge

```bash
bash slurm/submit_judge_panel_hpc.sh
```

To run only a subset:

```bash
bash slurm/submit_judge_panel_hpc.sh llama_8b_judge qwen_7b_judge
```

### Stricter masked control view

```bash
PAIRWISE_VIEW=canonical_masked bash slurm/submit_judge_panel_hpc.sh
```

### Single-judge full run

```bash
bash slurm/pre_cache_models.sh qwen_14b_judge
WAIT_FOR_JOB=1 bash slurm/submit_judge_hpc.sh qwen_14b_judge
```

---

## 11. Analysis

```bash
python cli.py analyze --judgments judgments/ --plans plans/ --pairs matched_pairs.json --output results/
```

---

## 12. Expected frozen study totals

* LLM arm: 256 plans
* Programmatic arm: 256 plans
* Total plans: 512
* Active judges: 4
* Pairwise calls: `250 pairs × 4 judges × 5 runs × 2 positions = 10,000`
* Soft-eval calls: `512 plans × 4 judges = 2,048`

---

## 13. Storage policy

Keep home usage below about **40 GB** before long runs.

Check usage with:

```bash
lquota
du -sh plans judgments results out err "${HOME}/hf_cache" "${HOME}/pip_cache" "${HOME}/torch_cache" "${HOME}/vllm_cache" 2>/dev/null
```

Delete in this order when tight:

1. old judge/source model caches already safely re-downloadable
2. stale smoke/pilot outputs
3. incomplete failed runs
4. old pip or conda caches if still necessary

Do not start long runs if you are already close to the 50 GB hard limit.

---

## 14. Monitoring

During jobs:

```bash
tail -f err/<job>.err
tail -f out/<job>.out
```

Job status:

```bash
sacct -j <JOBID> -P -o JobID,State,ExitCode,Reason,DerivedExitCode,MaxRSS,MaxVMSize,ReqMem
```

---

## 15. Recovery workflow

If code changed materially:

1. finish or cancel current jobs
2. rerun Gate 0
3. rerun preflight
4. rerun vLLM smoke
5. rerun a pilot judge
6. only then resume full generation or judging

If a sequential judge batch stops partway through:

* keep completed outputs
* inspect the failed judge log
* free space if needed
* rerun only the remaining judges

---

## 16. Why wrapped absolute-path `sbatch` is the default

On Bocconi, path resolution under Slurm can happen in spool directories rather than
your repo root. Using a wrapped `sbatch` with an absolute repo path makes the
execution context explicit and avoids relying on ambient shell state.

That is why this runbook standardizes on:

```bash
--wrap="cd ${REPO_ROOT} && bash slurm/<script>.sh"
```

rather than ambient `$PROJECT_ROOT` assumptions.

