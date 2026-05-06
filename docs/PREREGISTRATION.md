# Preregistration and Analysis Plan

## Research question

Controlling for deterministic structural plan quality, do LLM judges systematically favor LLM-generated training plans over programmatically generated plans, and does this effect vary by judge model family, model size, rubric, or whether the plan matches the judge's own family?

## Hypotheses

H1: LLM judges prefer LLM-source plans over matched programmatic controls more than 50% of the time after source-neutral structural matching.

H2: The preference may vary by judge family and model size.

H3: Self-family preference may occur when a judge family and source family overlap. Qwen-source plans can be tested with Qwen judges. Gemma-source plans can be tested with Gemma judges only if Gemma judges are present.

H4: Preference effects may differ by rubric. A style-mediated effect would be expected to appear more strongly in explanation-quality judgments than in purely structural coherence judgments.

## Experimental units

The primary unit is a matched pair containing one LLM-source plan and one programmatic plan from the same fixture cell and with similar source-neutral structural score. Each pair is evaluated by each judge model, for five repeated runs, in both AB and BA orders.

Full-study volume:

```text
250 matched pairs x 4 judge models x 5 repeats x 2 AB/BA positions = 10,000 pairwise judgments
```

The study must not be described as full unless this gate is satisfied.

## Candidate pool

The current operational pool contains 1024 candidate plans:

- 192 Qwen-source LLM plans
- 192 Gemma-source LLM plans
- 640 programmatic controls

All retained corpora must pass integrity checks for sidecars, JSON validity, plan length, prompt leakage, rest-active contradictions, and cell balance before matching.

## Matching score

The primary matching score is `structural_score_v1.0.0`. It is source-neutral and excludes human-facing presentation fields. The old TrailTraining quality score may be computed only as a diagnostic. It must not control primary matching unless it is proven invariant to presentation-only edits.

The structural score may use:

- session type sequence
- duration and weekly load
- active-day count
- rest-day count and spacing
- hard-day count and spacing
- long-run presence/count
- quality-session presence/count
- plan length
- athlete band, readiness, recovery capability, and race phase
- deterministic structural violations

The structural score must not use:

- title wording
- workout prose richness
- purpose prose richness
- citations
- claim attributions
- data-note verbosity
- explanation/rationale length
- source model name
- generation arm
- file naming artifacts

## Matching criteria

Primary matching should require same full fixture cell wherever possible:

- same fixture ID
- same athlete band
- same readiness
- same recovery capability
- same race phase
- same style
- same plan length

Within cell, matching uses structural score tolerance and weighted structural feature distance. The default primary tolerance is 2.0 structural-score points. This tolerance must be reported with mean, maximum, and p95 score gaps.

## Judge models

The active full-study judge panel must contain exactly four Qwen/Gemma local judge models. The intended current set is:

- `qwen_7b_judge`
- `qwen_14b_judge`
- `gemma_4b_judge`
- `gemma_12b_judge`

If fewer than four valid Qwen/Gemma judges are configured, the full study must stop rather than silently changing the design. No Llama judge should be configured or assumed for the current study.

## Evaluation protocol

Pairwise judging uses `compare_plans` from `soft_eval.py`. AB/BA order must be recorded. Source identity must be absent from the judge-facing artifacts. Metadata required for analysis is retained outside the judge prompt: source family, judge family, self-family match, fixture ID, structural score gap, order, run index, and judge model.

## Outcomes

Primary outcome: judge chooses the LLM-source plan over the matched programmatic control.

Secondary outcomes:

- per-judge LLM win rate
- per-source-family win rate
- per-judge-family win rate
- self-family match effect where defined
- order-bias audit
- score-gap sensitivity
- fixture/band subgroup analysis
- rubric-level results when soft-eval/per-rubric outputs exist

## Statistical analysis

Report descriptive estimates with confidence intervals before hypothesis interpretation. Use bootstrap confidence intervals for win-rate summaries. When model-based inference is used, prefer logistic models for binary choices, with clustering or random effects by matched pair when feasible.

Planned covariates include source family, judge family, self-family match, order, structural score gap, and fixture/cell. Ties should be reported separately and may be coded as 0.5 only for descriptive sensitivity summaries.

## Robustness checks

- AB/BA order-bias check
- structural score-gap sensitivity
- same-cell coverage by fixture
- source-family balance
- style/prose leakage audit
- duplicate/session-signature sensitivity
- temperature sensitivity in separate output directories

## Limitations to report

Equal structural score is not full equality. Programmatic plans may contain LLM-written explanation fields. Style normalization may remove real-world presentation variation. Local generation constraints and cache limitations affect external validity. The structural score itself is calibrated for this domain and should be treated as a control variable, not ground truth. Self-family analysis is valid only for represented source/judge family intersections.
