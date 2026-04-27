# PREREGISTRATION.md

**Status: FROZEN** — do not modify after the first generation run begins.

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
| LLM arm source models | meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen2.5-7B-Instruct |
| LLM exact-count rule | 16 plans per fixture per source model |
| Programmatic exact-count rule | 32 plans per fixture |
| Shared explainer model | `Qwen/Qwen2.5-3B-Instruct` |

### Matching

| Parameter | Value |
|---|---|
| Tolerance | `|Δscore| ≤ 1.0` |
| Target pairs | 250 |

### Judging

| Parameter | Value |
|---|---|
| Active judges | 4 (`llama_8b_judge`, `qwen_7b_judge`, `qwen_14b_judge`, `qwen_32b_judge`) |
| Pairwise calls | `250 × 4 × 5 × 2 = 10,000` |
| Soft-eval calls | `512 × 4 = 2,048` |
| Pairwise views | `raw_normalized` primary, `canonical_masked` control |
| Pilot gate | reject any judge with `|P(prefer position_a) − 0.5| > 0.2` on a 30-pair pilot |

### Exclusion criteria (frozen)

1. Schema-invalid responses are excluded from inference but retained for reliability accounting.
2. Unmatched plans are excluded from pairwise analysis but retained for soft-eval analysis.
3. Explainer mismatches are study-fatal and excluded from both arms.
4. Position-biased judges are excluded from H1/H2 primary analysis.

### Interpretation guardrail

The `raw_normalized` pairwise view captures structure plus presentation. The
`canonical_masked` view is the stricter control for testing whether any residual
preference survives stronger masking of surface cues.
