# Note 1 (orchestrator, 18:10Z)

Good start; the state-dtype check was the right first move. Three short points:
1. Your sweep uses 167-token prompts; the site's workload is 128 prompt tokens. Use the bench's own prompt construction (as `scripts/bench` builds the throughput point) or say in the report why 167 matches what the site ran. Same prompts file for every engine.
2. For TRT-LLM, record every non-default setting you had to choose (max_batch_size, max_seq_len, kv fraction, reuse off, scheduler policy) and compare pie against TRT-LLM's best working config, not a handicapped one.
3. If the slot fit at the declared context is the cause, the fix question is what context the bench declares and whether sizing slots to it is a bug or a policy; check what vLLM 0.32 does with the same max_model_len before choosing.
