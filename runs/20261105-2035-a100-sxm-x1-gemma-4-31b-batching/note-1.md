# Note 1 (orchestrator, 2026-11-05 23:10Z)

Good attribution. I checked round 1 against the raw files: they match (pie best 474.8 at c256, vLLM best 656.9 at c128, both at max_model_len 2048).

On the fix's scope:
1. #1368's cap in `crates/runtime/src/bootstrap.rs` applies to every seated-first profile (A100, A40, H100, H200, RTX PRO 6000, RTX 4090, RTX 5090), and this pod can measure only the A100. Prefer a change whose reach you can verify here, for example A100 profile settings (`seat_to_end`, and the cap made a profile setting if the runtime needs it). If you still change the runtime cap for everyone, explain in the report why it can't regress the other cards (#1368's own Gemma-4-26B numbers; the L4 c256 −13% in the ledger), and measure Gemma-4-26B-A4B on this A100 at c512 and c1024, the case #1368 was written for.
2. Another r2g run is on gpt-oss-120b × H100 at the same time. If you touch shared runtime admission, say so in status.md.
3. After the cap, seat_to_end's gate holds about 28 lanes where the pool fits 35 or more. That is the remaining −15%, and worth attributing before the hand-over, together with vLLM's 13.3k-token pool against pie's 10.7k.
