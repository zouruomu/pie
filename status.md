# status: 20261106-1706-h100-sxm-x1-qwen3-6-27b-batching

## Early verdict (Qwen3.6-27B, 1 round; Qwen3.8-27B and more rounds pending) — 18:50Z

**Red on main, and the cause is pie's state-slot sizing, not decode speed.** Same prompts (167-token chat-rendered, 2c requests at concurrency c, 128 out, greedy, ignore_eos), SM 1920-1980 MHz:

| tok/s | c64 | c128 | c256 | c512 | c1024 |
|---|---|---|---|---|---|
| pie main dc6282d76 | 1962 | **2197** | 2191 | 2086 | 1802 |
| vLLM 0.32.0 | 1756 | 2139 | **2159** | fails to start (336 mamba blocks) | fails |
| TRT-LLM 1.4.0rc2.post1, max_batch_size 128, KV frac 0.70 | – | 2674 | 2816 | **2848** | – |
| pie main, `--max-model-len 512` (experiment) | – | **2751** | 2477 | 2474 | – |

- pie vs TRT-LLM best-to-best: 2197 vs 2848 = −22.9% (TRT-LLM grid with max_seq_len 2048 at batch caps 160-224 still running: its best may rise).
- pie ≈ vLLM (+1.7%). No regression since the site's commit: ee9098f60 2177/2004 vs main 2180/2191 at c128/c256 (round 1).
- All my absolute numbers (pie, vLLM) sit ~12% under the site's: the site's request count / prompt length differ from mine (its harness is private). Ratios are what I compare.
- **Lanes:** pie decodes ~72 lanes a fire (runtime `wave avg active pipelines`) at ITL p50 28.5 ms; TRT-LLM up to 128 at 30 ms. Step time per lane is alike: the gap is lanes.
- **Why pie seats ~72-92:** its load fit (`Pools::fit_the_card`, crates/engine-cuda/src/store.rs:1130-1163, `balanced_state_slots`, crates/engine/src/fit.rs:152) cuts the recurrent-state slots until its kv pages hold *every slot at the declared context* (2048 tokens, 128 MiB of kv a sequence), while these sequences use ~295 tokens. c128: 256 slots asked, 92 served, kv pages for 47×2048 = 96k tokens of which ~27k are ever used. TRT-LLM sizes slots = max_batch_size and gives the rest to pages (`mamba_cache_manager.py:4190-4200`), then admits by each request's own prompt+max_tokens (GUARANTEED_NO_EVICT, `llm_args.py:4319`). With a 512-token declared context pie serves 134 slots, ~108 lanes a fire, 2751 tok/s (+25%), above TRT-LLM's c128.
- **Fairness:** TRT-LLM and pie both keep the GDN state in bf16 (pie `models/qwen_3.poem:1105-1106`; TRT-LLM ignores the checkpoint's `mamba_ssm_dtype: float32`, `config_utils.py:300-325`); vLLM keeps fp32. KV reuse off for TRT-LLM; same prompt ids; 128 tokens each (ignore_eos). TRT-LLM at its defaults does not start on 80 GB (max_batch_size 2048 → 160 GB of state slots; warmup at its default max_seq_len 65,536 OOMs); I set max_seq_len 2048 (= pie's/vLLM's max_model_len) and grid its batch cap.
- Second, smaller effect: pie sizes the fire's working set for max_forward_requests = c lanes (pie_bench), so the cache room and slots shrink as c grows (c64 92 → c1024 66 slots) though only ~90 can seat.

## Now
- Running: site-commit A/B round 2, then TRT-LLM grid (max_seq_len 2048, batch caps 160/192/224/128).
- Next: Qwen3.8-27B sweep (pie, TRT-LLM best config, vLLM); fix prototype for the slot/page split; nsys step tables.
- Hand-over ETA ~01:15Z.
