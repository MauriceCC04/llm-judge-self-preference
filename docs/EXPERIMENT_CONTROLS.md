# EXPERIMENT_CONTROLS.md

Research and implementation controls for `judge-bias-study`.

This document defines the rules that keep the study interpretable when using
`trailtraining` as a dependency and this repository as the experiment-control
layer.

## 1. Dependency policy

`trailtraining` is treated as a **pinned library dependency**.

We do **not** modify `trailtraining` for this study. Instead:

- generation control lives in this repository
- judge control lives in this repository
- provenance and matching discipline live in this repository
- sweep orchestration lives in this repository

This keeps the experimental logic explicit and versionable.

## 2. What is controlled here

This repository controls:

- fixture generation and frozen fixture usage
- plan generation orchestration
- provenance sidecars
- deterministic scoring
- greedy matched-pair construction
- pairwise and soft-eval harnesses
- judge resume behavior
- style leakage audits
- analysis loading and condition filtering
- temperature sweep orchestration

## 3. Temperature control model

Three temperatures are supported:

- `source_temperature`
- `explainer_temperature`
- `judge_temperature`

They correspond to three different causal surfaces:

### Source temperature
Changes the LLM arm generation process itself.

This may affect:

- structural choices
- verbosity
- lexical diversity
- explanation richness
- formatting regularity

### Explainer temperature
Changes the shared explainer stage.

This may affect:

- presentation style
- narrative detail
- rhetorical smoothness
- stylistic leakage across arms

### Judge temperature
Changes evaluator stochasticity while the plans remain fixed.

This is the cleanest way to study robustness of the judging process itself.

## 4. Baseline study settings

The frozen baseline study uses:

- `source_temperature = 0.7`
- `explainer_temperature = 0.0`
- `judge_temperature = 0.0`

These values are part of the baseline condition and must be visible in provenance
and judgment outputs.

## 5. Condition isolation rules

### Rule 1: one generation condition per plans directory
Never mix multiple generation conditions in one `plans/` directory.

### Rule 2: match within one generation condition
Run matching only on a single generation condition at a time.

### Rule 3: judge sweeps reuse fixed plans
When varying `judge_temperature`, reuse the same fixed plans and matched pairs.

### Rule 4: temperature must affect artifact identity
Temperature conditions must be encoded in:

- plan IDs
- provenance
- generation condition labels
- judgment rows
- output filenames
- artifact directories

## 6. Why this matters

Without these controls, the following failures become possible:

- resume collisions between distinct temperature runs
- mixed-condition matching
- provenance ambiguity
- accidental pooling of incomparable runs
- weakened causal interpretation

## 7. Matching interpretation

Deterministic score matching is necessary but not sufficient.

Even after matching on deterministic score, residual differences can remain in:

- verbosity
- wording
- explanation quality
- format regularity
- narrative specificity

Temperature changes can amplify these differences, especially for the LLM arm.

Therefore:

- baseline effects should use the frozen baseline condition
- source-temperature sweeps should be treated as generation-condition analyses
- judge-temperature sweeps should be treated as evaluator-robustness analyses

## 8. Pairwise and soft-eval views

Two pairwise comparison surfaces are supported:

- `raw_normalized`
- `canonical_masked`

The masked surface is especially useful when you want to reduce format-based or
layout-based cues.

Use masked comparison as a control condition, not as a replacement for the main
baseline unless pre-specified.

## 9. Analysis expectations

At minimum, downstream analysis should be able to stratify by:

- judge model
- judge family
- judge size
- pairwise view
- generation condition
- judge temperature
- LLM source temperature
- explainer temperature
- self-family match status

Temperature-aware analyses should report:

- win rate by condition
- confidence intervals
- coverage and matching diagnostics
- rubric-level changes where relevant
- style-leakage audit results where relevant

## 10. Recommended reporting language

Use language like:

- “baseline frozen study”
- “judge-temperature sensitivity analysis”
- “source-temperature sensitivity analysis”
- “shared-explainer temperature held fixed at 0.0”
- “trailtraining used as a pinned library dependency; experiment controls implemented in this repository”

Avoid language like:

- “temperature proves bias”
- “temperature only changes randomness”
- “matched score means plans are equivalent”

## 11. Minimal validation requirements

Before relying on temperature sweeps, validate that:

- generation provenance records the correct temperatures
- judgment records include `judge_temperature`
- resume keys separate distinct temperatures
- mixed-condition matching is blocked by default
- analysis loaders expose temperature columns for filtering and stratification

## 12. Summary

The study remains interpretable only if:

- `trailtraining` remains a stable dependency
- experiment control remains in this repository
- generation conditions remain isolated
- judge conditions remain explicit
- provenance remains complete
- analysis remains stratified where needed

These are not convenience rules. They are validity rules.
