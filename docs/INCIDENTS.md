# Incidents and Operational Decisions

## INC-022 - Legacy deterministic quality score failed as primary matching score

Date: 2026-05-06
Phase: matching
Severity: blocking

Symptoms: the old matcher produced approximately 30 pairs from the 1024-plan matching pool. LLM plans scored near 97-100 under the old score while most programmatic plans scored around 30, despite the programmatic corpus passing structural/artifact audits.

Root cause: the legacy matching path called TrailTraining's full deterministic quality score and the matching feature distance included presentation-derived terms such as narrative length and data-note counts. These are not valid primary controls for a study of judge preference because they may encode the very presentation confounds under investigation.

Fix: introduce `match/structural_score.py` and make structural score the primary matching score. Retain the old quality score only as a diagnostic. Remove narrative/prose/data-note features from weighted match distance.

Impact: full judging remains blocked until structural matching produces at least 250 same-cell matched pairs and the launch gate confirms at least 10,000 pairwise judgment records.

## INC-023 - 10,000 pairwise-judgment requirement clarified

Date: 2026-05-06
Phase: evaluation planning
Severity: validity-critical

The full study requires:

```text
250 matched pairs x 4 judge models x 5 repeated runs x 2 AB/BA positions = 10,000 pairwise judgments
```

Generated training plans are not evaluation documents. A 30-pair match would produce only 1,200 pairwise records and is invalid for the full study.

## INC-024 - Source masking and launch gate required before judging

Date: 2026-05-06
Phase: evaluation
Severity: validity-critical

Judge-facing artifacts must not include model names, source-family labels, generation arm labels, file names, or provenance sidecars. Source-family metadata is retained only in output metadata for analysis. The launch gate checks matching volume, judges, repetitions, AB/BA positions, source family coverage, structural score gaps, and masking before full pairwise evaluation.
