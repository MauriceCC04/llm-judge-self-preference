# Temperature Sweeps and Generation-Condition Rules

This document defines what may and may not be mixed in generation and matching artifacts.

## 1. Baseline condition

The current baseline generation condition is:

```text
gen_src_t070_exp_t000
```

Meaning:

- source temperature: 0.70
- explainer temperature: 0.00

Current retained baseline corpora:

```text
artifacts/gen_src_t070_exp_t000/full_qwen/plans
artifacts/gen_src_t070_exp_t000/full_gemma3/plans
artifacts/gen_src_t070_exp_t000/full_programmatic/plans
```

## 2. Allowed mixing

It is allowed to combine multiple source model families in one matching pool if all of the following are true:

- same fixture set
- same source temperature
- same explainer temperature
- same artifact validation gates
- same post-processing regime
- source family remains in provenance for analysis
- source family is hidden from judges

The current matching pool intentionally combines Qwen-source and Gemma-3-source LLM plans under the same baseline generation condition.

## 3. Disallowed mixing

Do not mix the following in one primary matching pool unless explicitly stratified and documented:

- different source temperatures
- different explainer temperatures
- different fixture versions
- different prompt versions
- different post-processing regimes
- pilot plans with full-run plans
- repaired artifacts with unrepaired artifacts unless the repair is deterministic and documented

## 4. Future temperature sweeps

Future temperature sweeps should be stored under separate roots, for example:

```text
artifacts/gen_src_t050_exp_t000/
artifacts/gen_src_t070_exp_t000/
artifacts/gen_src_t090_exp_t000/
```

Each sweep must have its own:

- plan directories
- provenance files
- validation audits
- matching pool
- matching audit
- judge outputs
- analysis outputs

Do not overwrite the current baseline.

## 5. Matching rule for sweeps

For the primary study, match only within the baseline condition unless the analysis plan is explicitly expanded.

For sweep analysis, compare conditions as sensitivity or exploratory analyses. Do not combine them into the primary effect estimate without modeling generation condition.

## 6. Documentation rule

Every generated plan must carry enough provenance to recover:

- source model
- explainer model
- source temperature
- explainer temperature
- fixture id
- athlete band
- readiness
- recovery capability
- race phase
- generation arm
- prompt/schema version if available
