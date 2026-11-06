# Batching part of the pie vs vLLM gap on A100 (Gemma-4-31B seated to the end past the five-sequence cap, a smaller working set)

DRAFT: numbers pending the 3-round table.

Gemma-4-31B bf16 on an A100 80 GB decoded ~21 sequences at once where vLLM 0.32.0 decodes ~33, at a 169-token prompt and 128 out. #1368's admission cap holds a seated-first pool to five sequences a declared-context sequence holds: at c512, 5 × 671 pages × 16 / 2048 = 26 processes, while the pool seats 35 of these 297-token sequences. And at the default 50% working set the fires (3,757 tokens) leave the cache rows 671 pages against vLLM's 11,929 tokens.

The A100 profile now seats to the end, as the A40, H100, H200 and RTX PRO 6000 do. A new profile decision, `engine.uncapped_admission` (default off, so every other card keeps the cap), lets that gate rather than the cap bound admission. And the working set is held to 40%: fires of 2,436 tokens, about vLLM's 2,496-token chunks, and 735 pages.

Gemma-4-31B c64-c1024 pending -> pending tok/s (vLLM pending; site best point pending); Gemma-4-26B-A4B c512 4.66k -> 5.16k (vLLM 4.62k); gpt-oss-120b and Qwen3-0.6B pending; Gemma-4-31B pbatch pending, raw pending.

Without the cap, seat_to_end alone (no cap and no seat_to_end loses 5% to suspended lanes) holds 35 lanes, the pool's lockstep bound; seat_to_end with the cap changes nothing. Greedy tokens: pending. Tests: runtime, pie-profile and engine-cuda chip pass; engine-cuda GPU suite pending.
