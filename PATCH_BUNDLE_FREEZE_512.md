# PATCH_BUNDLE_FREEZE_512

This branch contains the additive files for the 512-plan refreeze plus the exact intended replacement set for the mutable files that the connector could not update in place.

## Additive files already landed on this branch

- `generate/study_manifest.py`
- `judge/canonicalize.py`
- `tools/check_model_cache.py`
- `tools/verify_dependency_surface.py`
- `tools/repo_audit.py`
- `tools/render_frozen_docs.py`

## Intended in-place replacements

### `generate/constants.py`

- freeze totals at `256 / 256 / 512`
- keep `TARGET_PAIRS = 250`
- keep `PAIRWISE_N_RUNS = 5`
- keep `PAIRWISE_N_POSITIONS = 2`
- change default exact generation counts to `16` for the LLM arm and `32` for the programmatic arm
- add:
  - `PAIRWISE_VIEW_DEFAULT = "raw_normalized"`
  - `PAIRWISE_VIEW_CHOICES = ("raw_normalized", "canonical_masked")`

### `generate/exact_counts.py`

Replace the 500-plan top-up logic with the simpler frozen 512-plan logic:

- LLM: `16 plans × 8 fixtures × 2 source models = 256`
- programmatic: `32 plans × 8 fixtures = 256`

### `pyproject.toml`

Replace the placeholder trailtraining dependency pin with:

`trailtraining @ git+https://github.com/MauriceCC04/trailtraining.git@3e7f1793ca051ba1aae05f1714d594691202ad7e`

### `cli.py`

Add `--pairwise-view` to the `judge` subcommand and pass it through to `run_pairwise_harness`.

### `judge/harness.py`

Add canonical masked pairwise support by:

- importing `judge.canonicalize.canonicalize_pair_for_pairwise`
- adding helper `_prepare_pairwise_view(...)`
- adding `pairwise_view: str = PAIRWISE_VIEW_DEFAULT` to `run_pairwise_harness`
- recording `pairwise_view` in output rows

### `slurm/run_preflight.sh`

Add:

- `python tools/verify_dependency_surface.py`
- `import torch, vllm`
- `python -m vllm.entrypoints.openai.api_server --help`
- study manifest printout
- model cache checks via `tools/check_model_cache.py`

### `slurm/run_generation_hpc.sh`

Update comments and exact-profile expectations from `500` to `512` and add cache checks before launch.

### `slurm/run_judge_hpc.sh`

Preserve the existing walltime and style-gate logic, but add:

- `PAIRWISE_VIEW=${PAIRWISE_VIEW:-raw_normalized}`
- judge model cache check via `tools/check_model_cache.py`
- pass `--pairwise-view "${PAIRWISE_VIEW}"` into `python cli.py judge ...`

### `README.md`

Rewrite to match the frozen study:

- LLM arm: `256`
- Programmatic arm: `256`
- Total plans: `512`
- Target pairs: `250`
- Active judges: `4`
- Pairwise calls: `10,000`
- Soft-eval calls: `2,048`
- canonical HPC path: `HPC_SYNC_RUNBOOK.md`

### `HPC_SYNC_RUNBOOK.md`

Rewrite the canonical runbook so exact generation reflects:

- `128` LLM plans per source model
- `256` total LLM plans after both source jobs
- `256` programmatic plans
- `512` total plans
- optional `PAIRWISE_VIEW=canonical_masked`

### `HPC_RUNBOOK.md`

Demote to a legacy note that points users to `HPC_SYNC_RUNBOOK.md`.

### `PREREGISTRATION.md`

Freeze the scientific protocol to:

- `256 + 256 = 512` plans
- `250` target matched pairs
- `4` active judges
- `10,000` pairwise calls
- `2,048` soft-eval calls
- pairwise views: `raw_normalized` and `canonical_masked`

## Notes

The connector limitation here is specific to updates on already existing files. New files could be committed normally, which is why the new helper modules live directly on this branch while the mutable-file replacements are documented here.
