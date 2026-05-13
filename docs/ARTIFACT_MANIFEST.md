# Artifact Manifest

This manifest documents the key frozen artifacts for the Qwen/Gemma judge-bias study.

## Frozen root

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/
```

## Matching artifacts

| File | Purpose | Current status |
|---|---|---|
| `matched_pairs.json` | Frozen 250 matched LLM-vs-programmatic pairs | Complete |
| `matching_audit.json` | Matching diagnostics and balance | Complete |
| `matching_prefilter_audit.json` | Prefilter diagnostics | Complete |

Matching summary:

```text
matched pairs: 250
coverage_ok: true
mean structural score gap: 0.3076
p95 structural score gap: 2.0
max structural score gap: 2.0
```

## Evaluation manifest

| File/dir | Purpose | Current status |
|---|---|---|
| `eval_manifest/pairwise_eval_manifest.jsonl` | 10,000 judge-record manifest | Complete |
| `eval_manifest/judge_inputs/` | 10,000 canonical masked judge inputs | Complete |
| `eval_manifest/scrub_audit.json` | Source masking audit | Complete |
| `eval_manifest/pairwise_eval_manifest.jsonl.before_path_patch` | Historical backup | Keep as provenance or move to logs |

Manifest integrity:

```text
records: 10,000
unique record IDs: 10,000
pairs: 250
judges: 2,500 each
orders: 5,000 AB / 5,000 BA
runs: 2,000 each for run 0-4
```

## Primary pairwise outputs

Directory:

```text
judgments_eval_t000_scrubbed_v1_staged/
```

| File | Rows | Status |
|---|---:|---|
| `pairwise_gemma_4b_judge_canonical_masked_scrubbed_v1_t000.jsonl` | 2,500 | Complete |
| `pairwise_gemma_12b_judge_canonical_masked_scrubbed_v1_t000.jsonl` | 2,500 | Complete |
| `pairwise_qwen_7b_judge_canonical_masked_scrubbed_v1_t000.jsonl` | 2,500 | Complete |
| `pairwise_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl` | 2,500 | Complete |
| `pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl` | 10,000 | Complete |

Primary output result:

```text
LLM wins: 3,653
Programmatic wins: 6,347
LLM win rate: 36.53%
```

Checksum notes:

- Qwen 7B, Qwen 14B, Gemma 12B, and combined checksums were present and valid in the audit.
- Gemma 4B `.sha256` was empty in the uploaded audit. Regenerate before release.

## Explicit marker outputs

Directory:

```text
marker_eval_t000_scrubbed_v1_staged/
```

Current uploaded marker files:

| File | Rows | Status |
|---|---:|---|
| `marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl` | 2,500 | Complete |
| `marker_qwen_7b_judge_canonical_masked_scrubbed_v1_t000.jsonl` | 2,500 | Complete |
| `marker_gemma_12b_judge_canonical_masked_scrubbed_v1_t000.jsonl` | 0 / missing | Pending |
| `marker_gemma_4b_judge_canonical_masked_scrubbed_v1_t000.jsonl` | 0 / missing | Pending |

Current marker decision count:

```text
5,000 marker records × 9 markers = 45,000 explicit marker decisions
```

Full intended marker decision count after Gemma runs:

```text
10,000 marker records × 9 markers = 90,000 explicit marker decisions
```

Additional marker files:

| File | Purpose | Note |
|---|---|---|
| `failures_marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl` | Qwen 14B marker failures | 0 rows |
| `failures_marker_qwen_7b_judge_canonical_masked_scrubbed_v1_t000.jsonl` | Qwen 7B marker failures after repair | 0 rows |
| `failures_marker_qwen_7b_judge_canonical_masked_scrubbed_v1_t000.before_preferred_repair.jsonl` | Pre-repair failures | 43 rows; keep as provenance or move to logs |
| `missing_qwen_7b_marker_manifest.jsonl` | Temporary missing-record manifest | Remove or move after completion |

## Smoke-test artifacts

Directory:

```text
smoke_test/
```

Smoke-test outputs are provenance, not primary results. They may include nested-path artifacts from early marker runner debugging. Do not include smoke outputs in final primary or marker result tables.

## Hygiene before final public archive

Remove or move:

- `.DS_Store`
- `__MACOSX/`
- temporary missing manifests
- stale nested smoke/debug marker outputs
- malformed historical `PRIMARY_COMPLETION_STATUS.md` if still present

Regenerate:

- empty Gemma 4B `.sha256`
- top-level `SHA256SUMS.txt` excluding itself

Recommended clean zip command:

```bash
zip -r frozen_primary_v1_clean.zip artifacts/gen_src_t070_exp_t000/frozen_primary_v1 \
  -x "*/.DS_Store" \
  -x "__MACOSX/*" \
  -x "*/missing_*_marker_manifest.jsonl"
```

## Minimum integrity commands

Primary pairwise:

```bash
D="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t000_scrubbed_v1_staged"
P="$D/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl"
wc -l "$P"
jq -r '.record_id' "$P" | sort | uniq -d | wc -l
jq -r '.judge' "$P" | sort | uniq -c
jq -r '.order' "$P" | sort | uniq -c
jq -r '.run' "$P" | sort | uniq -c
```

Marker current Qwen-only:

```bash
D="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged"
for P in "$D"/marker_qwen_*_judge_canonical_masked_scrubbed_v1_t000.jsonl; do
  echo "=== $P ==="
  wc -l "$P"
  jq -r '.record_id' "$P" | sort | uniq -d | wc -l
  jq -r '.judge' "$P" | sort | uniq -c
  jq -r '.order' "$P" | sort | uniq -c
  jq -r '.run' "$P" | sort | uniq -c
  jq -r '.markers | length' "$P" | sort | uniq -c
  jq '[.markers[] | select(.normalization_warnings != null)] | length' "$P" | awk '{s+=$1} END {print "marker fields repaired:", s+0}'
done
```
