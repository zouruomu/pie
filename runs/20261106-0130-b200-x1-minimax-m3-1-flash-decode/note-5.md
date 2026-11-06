# Note 5 (orchestrator, 2026-11-06 05:52Z)

Early read of 98dccdc21 (the diff looks clean). Two things to cover before hand-over, so they don't cost a review round:
1. Reach: `mma_ranks` turns the cluster on wherever `blocks * 4 < sms`. With MiniMax's 4 kv heads on 148 SMs that is every fire of up to 9 rows, so c4 and c8 decode take the cluster path too, not only c1. Your curves (c4-c64) should show c4/c8 base vs fix explicitly, and the report should say where the path turns on and off (rows × kv heads), for MiniMax and for Qwen3.8-Flash-Next (2 kv heads: up to 18 rows).
2. Test coverage: `attn::selected` exercises the cluster walk only at Qwen3.8's head dim 256, but the shipped win is MiniMax's head dim 128 (G = 16 rows a kv head). If the test harness can take a second shape cheaply, add a 128-wide case with 16 query heads a kv head; otherwise say in the report why the 256 case covers the 128 instantiation.
