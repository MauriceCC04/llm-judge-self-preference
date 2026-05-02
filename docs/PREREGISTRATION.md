# PREREGISTRATION.md

**Status: FROZEN** — the estimand and primary design are frozen. Clarifications
below document retained-corpus eligibility and implementation discipline learned
during the first real generation/debugging cycle. They do not change the core
research question.

---

## Study title

LLM-judge self-preference on training plans: a controlled experiment on
Bocconi Jupiter I HPC.

## Research question

Controlling for deterministic plan quality, do LLM judges systematically favour
LLM-generated training plans over programmatically generated ones, and does this
effect vary by judge model family and scale?

---

## Study design parameters (frozen)

### Fixtures

| Parameter | Value |
|---|---|
| Number of fixture bundles | 8 |
| Fixture generation | deterministic |

### Generation

| Parameter | Value |
|---|---|
| LLM arm | 256 plans |
| Programmatic arm | 256 plans |
| Total plans | 512 |
| LLM arm source models | `Qwen/Qwen2.5-7B-Instruct`, `google/gemma-3-4b-it` |
| LLM exact-count rule | 16 plans per fixture per source model |
| Programmatic exact-count rule | 32 plans per fixture |
| Shared explainer model | `Qwen/Qwen2.5-3B-Instruct` |
| Baseline source temperature | `0.7` |
| Baseline explainer temperature | `0.0` |

### Matching

| Parameter | Value |
|---|---|
| Tolerance | `|Δscore| ≤ 1.0` |
| Target pairs | 250 |
| Matching discipline | one generation condition at a time |

### Judging

| Parameter | Value |
|---|---|
| Active judges | 4 (`qwen_7b_judge`, `qwen_14b_judge`, `gemma_4b_judge`, `gemma_12b_judge`) |
| Pairwise calls | `250 × 4 × 5 × 2 = 10,000` |
| Soft-eval calls | `512 × 4 = 2,048` |
| Pairwise views | `raw_normalized` primary, `canonical_masked` control |
| Pilot gate | reject any judge with `|P(prefer position_a) − 0.5| > 0.2` on a 30-pair pilot |
| Baseline judge temperature | `0.0` |

## Exclusion criteria (frozen, clarified)

1. Schema-invalid responses are excluded from inference but retained for reliability accounting.
2. Unmatched plans are excluded from pairwise analysis but retained for soft-eval analysis.
3. Explainer mismatches are study-fatal and excluded from both arms.
4. Position-biased judges are excluded from H1/H2 primary analysis.
5. Saved plans with placeholder leakage (for example `>{signal_id`) are excluded from all retained-corpus analyses.
6. Saved plans with human-facing contradictions (for example a day titled `Rest day` with `session_type != rest` or `duration_minutes > 0`) are excluded from all retained-corpus analyses.
7. Exact text duplicates are collapsed to a single retained representative before matching.
8. Exact session-signature duplicates are collapsed to a single retained representative before matching within the active generation condition / cell.

## Retained-corpus clarification

Equal deterministic score is necessary but **not sufficient** to establish a
clean comparison set.

The retained corpus used for matching and judging must therefore satisfy both:

- deterministic validity / score-based comparability
- artifact-integrity and deduplication rules

This means matching is run on the **filtered retained corpus**, not on the raw
pre-filter generation pool.

## Human audit clarification

A lightweight human audit is required before the full judging panel is launched.
This audit is intended to catch study-invalid artifacts that survive schema and
deterministic checks but remain implausible or contradictory to a human reader.

Recommended minimum:

- targeted review of suspicious duplicate groups
- stratified spot-checking across the active fixture cells
- explicit pass / questionable / fail labeling during review

## Interpretation guardrail

The `raw_normalized` pairwise view captures structure plus presentation. The
`canonical_masked` view is the stricter control for testing whether any residual
preference survives stronger masking of surface cues.

## Post-freeze implementation note

During the initial HPC debugging cycle, the most important operational lesson
was that **artifact integrity and corpus filtering must happen before matching**.
This note clarifies implementation discipline and retained-corpus eligibility; it
does not alter the estimand itself.
