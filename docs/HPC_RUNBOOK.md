# HPC Runbook

This runbook is the canonical operational guide for the current 1024-candidate-pool study state. It supersedes earlier frozen-512 commands.

## 0. Current artifact layout

Current retained corpora:

```text
artifacts/gen_src_t070_exp_t000/
  full_qwen/plans/
  full_gemma3/plans/
  full_programmatic/plans/
  matching_pool/plans/
  matching_pool/matched_pairs.json
  matching_pool/matching_audit.json
  matching_pool/matching_prefilter_audit.json
```

Expected counts before matching:

```text
full_qwen:         192 plans, 192 provenance, 32 cells, 6 per cell
full_gemma3:       192 plans, 192 provenance, 32 cells, 6 per cell
full_programmatic: 640 plans, 640 provenance, 32 cells, 20 per cell
matching_pool:     1024 plans total
```

The full judge evaluation must not launch until a valid structural-score match produces at least 250 matched pairs.

## 1. Golden Python environment pattern

Do not rely on plain `conda activate` in the Bocconi interactive shell. It may fail with:

```text
CondaError: Run 'conda init' before 'conda activate'
```

Use the environment Python directly and explicitly set `PYTHONPATH` so both repositories are importable.

```bash
cd /mnt/beegfsstudents/home/<USER_ID>/llm-judge-self-preference

export REPO_ROOT=/mnt/beegfsstudents/home/<USER_ID>/llm-judge-self-preference
export TRAILTRAINING_REPO=/mnt/beegfsstudents/home/<USER_ID>/trailtraining
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"
export PY=/home/<USER_ID>/.conda/envs/judge-bias/bin/python

$PY - <<'PY'
import sys
import pydantic
import trailtraining
print('python', sys.executable)
print('pydantic', pydantic.__version__)
print('trailtraining', trailtraining.__file__)
PY
```

Expected shape:

```text
python /home/<USER_ID>/.conda/envs/judge-bias/bin/python
trailtraining /mnt/beegfsstudents/home/<USER_ID>/trailtraining/src/trailtraining/__init__.py
```

If `trailtraining` cannot be imported, deterministic scoring and matching will fail.

## 2. Hugging Face cache and quota rules

Known issues encountered:

- Gemma models are gated. You must accept terms and set a valid `HF_TOKEN` before caching.
- `HF_HUB_DISABLE_XET=1` should be exported.
- Quota can be tight. Do not keep unrelated model families cached.
- For LLM generation, cache exactly the source model and the shared explainer.
- For programmatic generation, cache only the shared explainer.
- For judge runs, cache only the active judge model.
- Do not set cleanup flags that delete models needed by subsequent jobs unless that is intentional.

Recommended cache environment:

```bash
export HF_HOME=/mnt/beegfsstudents/home/<USER_ID>/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
lquota
```

Model-cache verification:

```bash
$PY tools/check_model_cache.py "Qwen/Qwen2.5-3B-Instruct"
$PY tools/check_model_cache.py "Qwen/Qwen2.5-7B-Instruct"
$PY tools/check_model_cache.py "google/gemma-3-4b-it"
```

## 3. Corpus integrity audit

Run before constructing the matching pool.

```bash
cd /mnt/beegfsstudents/home/<USER_ID>/llm-judge-self-preference

export REPO_ROOT=/mnt/beegfsstudents/home/<USER_ID>/llm-judge-self-preference
export TRAILTRAINING_REPO=/mnt/beegfsstudents/home/<USER_ID>/trailtraining
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"
export PY=/home/<USER_ID>/.conda/envs/judge-bias/bin/python

$PY - <<'PY'
import json
from pathlib import Path
from collections import Counter

corpora = {
    'qwen': Path('artifacts/gen_src_t070_exp_t000/full_qwen/plans'),
    'gemma3': Path('artifacts/gen_src_t070_exp_t000/full_gemma3/plans'),
    'programmatic': Path('artifacts/gen_src_t070_exp_t000/full_programmatic/plans'),
}

for name, d in corpora.items():
    plans = sorted(p for p in d.glob('*.json') if not p.name.endswith('.provenance.json'))
    provs = sorted(d.glob('*.provenance.json'))
    by_cell = Counter()
    by_band = Counter()
    bad = []
    missing_sidecars = []

    for p in plans:
        sidecar = p.with_name(p.name + '.provenance.json')
        if not sidecar.exists():
            missing_sidecars.append(p.name)
            continue
        try:
            obj = json.loads(p.read_text())
            prov = json.loads(sidecar.read_text())
            days = obj.get('plan', {}).get('days', [])
            if len(days) != 7:
                bad.append((p.name, f'{len(days)} days'))
            by_cell[prov.get('fixture_id')] += 1
            by_band[prov.get('athlete_band')] += 1
        except Exception as e:
            bad.append((p.name, str(e)))

    expected_per_cell = 20 if name == 'programmatic' else 6
    non_expected = {k: v for k, v in sorted(by_cell.items()) if v != expected_per_cell}

    print(f'\n=== {name} ===')
    print('plans', len(plans))
    print('provenance', len(provs))
    print('cells', len(by_cell))
    print('by_band', dict(sorted(by_band.items())))
    print('cell_count_min_max', (min(by_cell.values()), max(by_cell.values())) if by_cell else None)
    print('non_expected_cells', non_expected)
    print('missing_sidecars', missing_sidecars[:10])
    print('bad', bad[:10])
PY
```

Expected:

```text
qwen:         192 plans, 32 cells, 6 per cell
gemma3:       192 plans, 32 cells, 6 per cell
programmatic: 640 plans, 32 cells, 20 per cell
```

## 4. Matching-pool construction

The matcher expects one plans directory. Construct it from the three validated corpora.

```bash
cd /mnt/beegfsstudents/home/<USER_ID>/llm-judge-self-preference

POOL=artifacts/gen_src_t070_exp_t000/matching_pool/plans
rm -rf "$POOL"
mkdir -p "$POOL"

$PY - <<'PY'
import json
import shutil
from pathlib import Path

sources = [
    Path('artifacts/gen_src_t070_exp_t000/full_qwen/plans'),
    Path('artifacts/gen_src_t070_exp_t000/full_gemma3/plans'),
    Path('artifacts/gen_src_t070_exp_t000/full_programmatic/plans'),
]

pool = Path('artifacts/gen_src_t070_exp_t000/matching_pool/plans')
pool.mkdir(parents=True, exist_ok=True)

copied = 0
for src in sources:
    for plan_path in sorted(src.glob('*.json')):
        if plan_path.name.endswith('.provenance.json'):
            continue
        prov_path = plan_path.with_name(plan_path.name + '.provenance.json')
        if not prov_path.exists():
            raise SystemExit(f'Missing provenance for {plan_path}')
        dst_plan = pool / plan_path.name
        dst_prov = pool / prov_path.name
        if dst_plan.exists() or dst_prov.exists():
            raise SystemExit(f'Collision in matching pool: {plan_path.name}')
        shutil.copy2(plan_path, dst_plan)
        prov = json.loads(prov_path.read_text())
        prov['plan_path'] = str(dst_plan)
        dst_prov.write_text(json.dumps(prov, indent=2, ensure_ascii=False))
        copied += 1

print('copied_plans', copied)
print('pool', pool)
PY
```

Expected:

```text
copied_plans 1024
```

## 5. Known invalid matching result

The old matching command completed but yielded only 30 pairs:

```text
Pairs yielded: 30 (target: 256)
Coverage ratio: 0.117
Coverage OK: False
```

This matched set is invalid for full judging.

The issue was not invalid plans or duplicate filtering. The issue was severe score non-overlap under the old quality score:

```text
Qwen median score:       about 100
Gemma 3 median score:    about 100
Programmatic median:     about 30
```

Relaxing same-bin matching did not solve this. Do not use this old-score result for judging.

## 6. Required matching implementation

Before rerunning final matching, implement source-neutral structural scoring. See `MATCHING_STRUCTURAL_SCORE_RUNBOOK.md`.

Full matching may proceed only if:

```text
matched pairs >= 250
preferably matched pairs >= 256
coverage across all 32 fixture cells is documented
structural score gap mean/max/p95 are acceptable
Qwen and Gemma source families are represented
style/leakage gate passes
```

## 7. Safe SLURM pattern for structural matching

```bash
cd /mnt/beegfsstudents/home/<USER_ID>/llm-judge-self-preference
mkdir -p out err

sbatch \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
  --time=00:45:00 \
  --chdir=/mnt/beegfsstudents/home/<USER_ID>/llm-judge-self-preference \
  --output=out/match_pool_structural_%j.out \
  --error=err/match_pool_structural_%j.err \
  --wrap='
set -euo pipefail
export REPO_ROOT=/mnt/beegfsstudents/home/<USER_ID>/llm-judge-self-preference
export TRAILTRAINING_REPO=/mnt/beegfsstudents/home/<USER_ID>/trailtraining
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"
export PY=/home/<USER_ID>/.conda/envs/judge-bias/bin/python
cd "$REPO_ROOT"
$PY - <<PY
import sys, pydantic, trailtraining
print("python", sys.executable)
print("pydantic", pydantic.__version__)
print("trailtraining", trailtraining.__file__)
PY
$PY cli.py match \
  --plans artifacts/gen_src_t070_exp_t000/matching_pool/plans \
  --output artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json \
  --allow-mixed-generation-conditions
'
```

## 8. 10,000 evaluation-document launch gate

Full judge evaluation must assert:

```text
n_pairs >= 250
n_orders == 2
n_runs == 5
n_judge_models == 4
n_eval_documents >= 10000
```

Formula:

```text
250 matched pairs x 2 orders x 5 runs x 4 judge models = 10,000 evaluation documents
```

Preferred:

```text
256 matched pairs x 2 orders x 5 runs x 4 judge models = 10,240 evaluation documents
```

Pilot jobs may use fewer pairs but must write to pilot-specific directories and must not be reported as the full 10,000-document study.

## 9. Common failure modes and fixes

### `ModuleNotFoundError: No module named 'pydantic'`

Wrong Python environment. Use:

```bash
export PY=/home/<USER_ID>/.conda/envs/judge-bias/bin/python
```

### `ModuleNotFoundError: No module named 'trailtraining'`

Missing sibling repo in `PYTHONPATH`. Use:

```bash
export TRAILTRAINING_REPO=/mnt/beegfsstudents/home/<USER_ID>/trailtraining
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"
```

### `Model not found in cache`

The model was deleted or not cached. Re-cache exactly the needed model(s). Check quota before caching.

### Gemma 401/gated repo

Set `HF_TOKEN` and make sure the Hugging Face account has accepted the model terms.

### Full matching yields ~30 pairs

Do not generate more blindly and do not loosen tolerance. Implement structural scoring and rerun matching diagnostics.
