# Primary pairwise judging completion status

Status: COMPLETE

Primary input:
- frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl
- canonical_masked_scrubbed_v1 judge inputs

Primary output:
- judgments_eval_t000_scrubbed_v1_staged/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl

Expected design:
- 250 matched pairs
- 4 Qwen/Gemma judge models
- 5 repeated runs
- 2 AB/BA orders

Expected records:
- 250 x 4 x 5 x 2 = 10,000

Observed records:
- 10,000 total
- 0 duplicate record_id
- 2,500 per judge
- 5,000 AB / 5,000 BA
- 2,000 per run index 0-4

Completed judge files:
- pairwise_gemma_4b_judge_canonical_masked_scrubbed_v1_t000.jsonl
- pairwise_gemma_12b_judge_canonical_masked_scrubbed_v1_t000.jsonl
- pairwise_qwen_7b_judge_canonical_masked_scrubbed_v1_t000.jsonl
- pairwise_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl

Notes:
- Qwen 14B AWQ required quantization argument normalization from awq_int4 to awq.
- Qwen 14B required TRAILTRAINING_STRUCTURED_MAX_TOKENS to be capped below the 4096 context length.
- Qwen 14B was resumed after a partial run; final record_id uniqueness is 0 duplicates.
