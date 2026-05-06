# 10,000 Evaluation-Document Launch Gate

This file defines the hard launch rule for the full judge-evaluation sweep.

## Required formula

The primary pairwise evaluation design requires at least 10,000 judge-facing evaluation documents.

The formula is:

```text
matched pairs x left-right orders x repeated runs x judge models
```

For this study:

```text
left-right orders = 2
repeated runs = 5
judge models = 4
```

Therefore:

```text
250 matched pairs x 2 orders x 5 runs x 4 judge models = 10,000 evaluation documents
```

Preferred:

```text
256 matched pairs x 2 orders x 5 runs x 4 judge models = 10,240 evaluation documents
```

## Current status

The generated corpora are large enough in principle:

- 384 LLM-source plans.
- 640 programmatic candidate plans.
- 1024 total matching-pool plans.

The old-score matched set contains only 30 pairs:

```text
30 x 2 x 5 x 4 = 1,200 evaluation documents
```

That is not sufficient and must not be used for the full study.

## Required assertion before full judging

A full judge launcher must compute and print:

```text
n_pairs
n_orders
n_runs
n_judge_models
n_eval_documents
```

It must refuse full launch unless:

```text
n_pairs >= 250
n_orders == 2
n_runs == 5
n_judge_models == 4
n_eval_documents >= 10000
```

## Pilot exception

A pilot run may use fewer pairs, fewer judges, fewer runs, or one order. It must:

- write to a pilot-specific directory
- print that it is a pilot
- never overwrite full-run outputs
- never be reported as satisfying the 10,000-document criterion

## Required metadata

Each judge-facing evaluation document must have metadata linking it to:

- matched pair id
- fixture cell
- LLM plan id
- programmatic plan id
- source family metadata hidden from judge
- judge model
- run index
- left/right order
- rubric or overall criterion
- prompt version
- structural score and score gap

## Reporting rule

The paper/report must distinguish:

- generated plans
- retained candidate plans
- matched pairs
- judge-facing pair/order documents
- repeated-run documents
- final valid judgment records after schema filtering

Do not report the study as satisfying the 10,000-document criterion unless at least 10,000 judge-facing evaluation documents were actually attempted under the accepted matched-pair set.
