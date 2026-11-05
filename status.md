# Status: 20261105-2035-a100-sxm-x1-gemma-4-31b-batching

**Phase:** attribution: cap experiments done, seat_to_end trace running
**UTC:** 2026-11-05 22:55

## Preliminary verdict (1 round so far; rounds 2-3 running)

The gap is real on current main (e4bb499):
- pie c512 453.8 tok/s (site's pie point: c512, 465.76). TTFT p50 132 s, ITL p50 46.2 ms: it reproduces the site row.
- vLLM 0.32.0, round 1: c64 601, **c128 657**, c256 625, c512 629, c1024 597 tok/s (site: 569 at c1024).
- Round 1, pie (main e4bb499, engine sha256 c795a06a): c128 458, c256 475, c512 454, c1024 471 tok/s. c64 is to be rerun (its file was overwritten, see Notes).
- **Verdict: the gap is real and larger than the site's: pie's best 475 (c256) against vLLM's best 657 (c128), -28%.** From c128 up pie flattens at ~21 decoding. (Correction: an earlier line here gave a pie c64 of 601. That number was the overwritten file's, i.e. vLLM's. pie's c64 is pending.)
- Rounds 2-3 for spread are queued after the experiments (~23:05Z onward).

How many sequences each engine decodes (tok/s x ITL p50):
- vLLM: 31-35 at 51-53 ms. Its KV pool is 13,307 tokens (11.3 GiB) at c128 and 10,889 at c1024: CUDA graphs grow with max_num_seqs.
- pie: 21 at 46 ms.
  - Its pool at c512 is 671 pages of 16 = 10,736 tokens (9,280 MiB, kv + window rows), enough to seat ~35 sequences of 297 tokens (19 pages each).
  - But #1368's admission cap (`crates/runtime/src/bootstrap.rs:318-340`) admits 5 x 671 x 16 / 2048 = **26 processes**. The cap binds before the pool does.

pie's memory at c512, from the arm-trace log (results/diag): weights 57.5 GiB, context before load 0.93 GiB, activation arena 1.34 GiB (fires cut 8192 -> 3757 tokens by working_set_percent 50), inputs 0.66 GiB, graph bodies 0.54 GiB, kernel scratch 0.26 GiB. That leaves 9.58 GiB of room, of which the pool takes 9.06 GiB.

## Cap experiments (c512, results/exp, exp engine = main + env-set cap multiplier)

| run | admitted | tok/s | ITL p50 / p99 ms |
|---|---|---|---|
| base (x5) | 26 | 453.8 | 46.2 / 48.6 |
| x5 + seat_to_end | 26 | 453.7 | 46.2 / 48.8 |
| no cap | 512 | 433.2 | 0.09 / 1127 (lanes suspended and replayed) |
| x7 | 36 | 548.3 | 49.2 / 317 |
| x7 + seat_to_end | 36 | 559.5 | 49.3 / 52.3 |
| no cap + seat_to_end | 512 | 559.4 | 49.3 / 51.8 |

The cap is what holds pie at ~21 lanes. seat_to_end gates cleanly without it (+23%). But it still holds ~28 lanes where 671 pages fit ~35-44 lanes of 19 pages. Tracing its gate (learned footprint against kv_total) now.

## Running
- Queue q1 (scripts/q1.sh): pie round 1 over c64-c1024, then vLLM and pie rounds 2-3, alternating.
- An experiment engine is building in a separate worktree (/root/r2g/pie-exp, not the fix): it reads the admission cap's multiplier from an env var, so the cap can be swept (x5, x7/x8, off) with and without `seat_to_end`.

## Notes
- 20:49Z the 300 GB disk filled (gpt-oss-120b's metal/ and original/ copies). Freed it; lost ~15 min of runs.
- At 21:42Z I edited bench.sh in place while a pie c64 run was executing it. bash resumed at its old byte offset and ran vLLM under pie's c64-r1 label. That file is in results/bad/; no other run was affected (the engine field and sha256 were checked in every JSON).
- `--pretokenized-prompts` needs transformers in pie's venv; it was installed after the first pie attempt failed.

ETA: early verdict (3 rounds) ~22:45Z. Hand-over ~04:30Z.
