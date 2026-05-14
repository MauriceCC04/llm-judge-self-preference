# Temperature Sweeps and Evaluation Conditions

This document records the temperature conditions used in the completed study and how future temperature sensitivity checks should be handled.

## Final primary condition

The completed primary and marker-level results use:

```text
JUDGE_TEMPERATURE=0.0
```

This is encoded in output names as:

```text
t000
```

Primary pairwise outputs:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t000_scrubbed_v1_staged/
```

Marker outputs:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/
```

Derived results:

```text
results/primary_t000_scrubbed_v1/
```

## Completed temperature status

The finished report should treat `t000` as the primary completed evaluation condition.

Temperature sweeps beyond `t000` were not required to complete the current report. If future sweeps are performed, they should be treated as sensitivity analyses and stored in separate directories.

## Do not mix conditions

Never mix generation/evaluation temperature conditions in the same output directory.

Use separate names such as:

```text
judgments_eval_t020_scrubbed_v1_staged/
marker_eval_t020_scrubbed_v1_staged/
results/primary_t020_scrubbed_v1/
```

The current completed outputs are strictly `t000`.

## Recommended future sweep design

If future time permits, run a smaller temperature sensitivity check rather than a full rerun:

- Use the same frozen matched pairs and judge inputs.
- Use the same judge set.
- Keep AB/BA order and repeated runs.
- Store outputs in temperature-specific directories.
- Do not combine temperature conditions in the primary analysis.

Possible sensitivity temperatures:

```text
0.0 primary
0.2 sensitivity
0.7 stress/sensitivity only
```

Because the current primary results show strong position effects, any temperature sensitivity analysis must include AB/BA pair-run consistency, not just row-level win rates.

## Required gates for any future temperature run

Before launching a full future temperature condition:

- Frozen matched pairs exist and have at least 250 pairs.
- Manifest has exactly 10,000 records for the full design.
- Exactly 4 Qwen/Gemma judges are configured.
- Both AB and BA orders are present.
- Source masking audit passes.
- Output directory is temperature-specific.
- Existing primary `t000` outputs are not overwritten.

## Final report wording

Use this wording for the completed report:

```text
The primary analyses use deterministic judge settings at temperature 0.0. Additional temperature sweeps were not included in the completed primary report. All reported primary and marker-level results are from the frozen `t000` evaluation condition.
```
