# status: 20261106-1706-h100-sxm-x1-qwen3-6-27b-batching

- 18:04Z phase: reproduce (Qwen3.6-27B). Running one serial GPU queue: vLLM q36 sweep -> TRT-LLM config grid (max_seq_len 2048) -> pie `--max-model-len 512` experiment -> site commit ee9098f60 vs main A/B.
- **pie main dc6282d76, Qwen3.6-27B, 1 round** (2c requests, c concurrency, 167-token prompts, 128 out): c64 1962, c128 **2197**, c256 2191, c512 2086, c1024 1802 tok/s. Site row (ee9098f60) was 2489; checking for a regression.
- **TRT-LLM at its defaults does not start on this card** (max_batch_size 2048 needs 160 GB of state slots; with smaller batch caps its warmup OOMs on a 65,536-token sequence, its default max_seq_len). With max_batch_size 128, KV fraction 0.70: c128 2674, c256 2816, c512 2848 tok/s (ITL p50 30 ms). Grid with max_seq_len 2048 (pie's and vLLM's declared context) running.
- **State dtype:** pie (models/qwen_3.poem:1105-1106) and TRT-LLM (config_utils.py:300-325) both keep the GDN state in bf16; vLLM 0.32 keeps it fp32 (the checkpoint's mamba_ssm_dtype). So pie vs TRT-LLM is like for like on state precision.
- **Lead:** pie seats ~66-92 state slots (~72 active lanes a fire at c128). pie's load-time fit cuts the slots so that its kv pages hold every slot at the declared context (2048), \`Pools::fit_the_card\` crates/engine-cuda/src/store.rs:1130-1163 + \`balanced_state_slots\` crates/engine/src/fit.rs:152; the sequences use ~295 tokens. TRT-LLM reserves each request's prompt+max_tokens (GUARANTEED_NO_EVICT). Sizing this next.
- Early verdict ETA ~19:30Z; hand-over ETA ~01:15Z.
