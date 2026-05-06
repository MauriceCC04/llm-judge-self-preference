# Temperature Sweeps and Sensitivity Runs

The primary full-study result should use one clearly designated judge temperature, normally `0.0`, and write to a temperature-specific output directory. Sensitivity runs must not be mixed into the primary JSONL files.

## Directory convention

```text
artifacts/gen_src_t070_exp_t000/matching_pool/
  matched_pairs.json
  judgments_eval_t000/        # primary, judge temperature 0.0
  judgments_eval_t030/        # sensitivity, judge temperature 0.3
  judgments_eval_t070/        # sensitivity, judge temperature 0.7 if used
  results_eval_t000/
  results_eval_t030/
```

The generation temperature condition is encoded in the artifact root (`gen_src_t070_exp_t000`). Do not mix generation-temperature roots in one matching or judging directory.

## Primary run

```bash
JUDGE_TEMPERATURE=0.0 \
JUDGMENTS_DIR=artifacts/gen_src_t070_exp_t000/matching_pool/judgments_eval_t000 \
sbatch slurm/run_judge_hpc.sh
```

## Sensitivity run

```bash
JUDGE_TEMPERATURE=0.3 \
JUDGMENTS_DIR=artifacts/gen_src_t070_exp_t000/matching_pool/judgments_eval_t030 \
sbatch slurm/run_judge_hpc.sh
```

## Required checks

For each temperature condition, keep separate:

- pairwise JSONL outputs
- schema failures
- analysis summaries
- manifest/audit files

A sensitivity temperature run may use the same matched pairs, but it must be labeled as sensitivity and must not contaminate the primary 10,000-judgment result unless explicitly designated before analysis.
