# Status: 20261106-0130-b200-x1-minimax-m3-1-flash-decode

UTC: 2026-11-06 06:50Z
Phase: GPU suite running on 3da03d2c1 (started 06:50Z, ~1.5 h). Done on the final build: c1 A/B (3 rounds), curves both cuts, cross-model. After the suite: pf32 (prompt file was 2 short), pf8 x3 on cut B, nsys at c32, rebase onto upstream main + one A/B round. ETA hand-over ~09:15Z.

## Verdict
- The gap is real on B200, but the site's competitor numbers can't be this checkpoint: no released vLLM (0.32.0, 0.33.0) or SGLang (0.5.22) loads MiniMax-M3.1-Flash's NVFP4 experts. Fair same-weights comparison: the same 60-layer, 16-expert cut of MiniMax-M3-MXFP8 (cut B), which they do load.
- Fix (one commit): MiniMax's decode selected attention over a cluster of 8 blocks a kv head (it ran on 4 blocks: one a kv head). Final engine, 3 rounds, busy SM 1965 MHz:
  - cut A (M3.1-Flash, the site's checkpoint): 8.976 -> 8.620 ms ITL p50, 111.4 -> 116.0 tok/s (+4.1%);
  - cut B (M3-MXFP8): 9.623 -> 9.286 ms, 103.9 -> 107.7 tok/s (+3.6%); vLLM 0.32.0 same session 6.925 ms: pie behind 28.0% -> 25.4%.
- Tokens: cut B identical 8/8 x 256; cut A first diverges at token 1-8 on 6/8 at bf16 near-ties; teacher-forced decode-position KL below the scalar selected walk's (the accepted reference) on all 8 probed requests.
- Fix 2 (q/k norm + RoPE + KV append, ~12.7-14.4 us a layer in two 1-CTA epilogue slices) measured by a reverted prototype at ~4% more, but its shippable form is skeleton/compiler work and the quick form revives a rule CUDA retired: handed over as attribution. Expert glue likewise.

## Review round (independent subagent, results in /root/r2g/review/review-1.md), acknowledged and acted on
- Blockers fixed: cargo fmt (selected.rs); the unit test never folded two non-empty slices -> new test `a_clusters_slices_fold_into_what_one_block_walks` (MiniMax's shape, 1100 keys, every rank walks keys); an unweighted fold now fails both tests (results/tests/kernels-selected-mutation-unweighted-fold.log).
- Code: the cluster rule now halves the ranks while the clustered blocks exceed a quarter of the SMs (the first build lost 1% at c8 and gained nothing at c4: results/raw/curve-fix1), plus an sm_90 check. Amended into the one commit (3da03d2c1), force-pushed.
- Write-up: every number recomputed from the JSONs' exact ITLs (results/scripts/numbers.py); the scalar-walk comparison moved to decode positions only; near-tie claims limited to the 3 probed divergences; profile statements, file:line refs, prefill wording, nsys provenance and the prototype's 1792 MHz caveat corrected.

## Notes acknowledged
- Note 5 (05:52Z): both covered by the review round. (1) Reach: the final rule halves the ranks while the clustered blocks exceed a quarter of the SMs; on 148 SMs MiniMax (4 kv heads) 1 row 8 ranks, 2 rows 4, 3-4 rows 2, 5+ off; Qwen3.8-Flash-Next (2 kv heads) 1-2 rows 8, 3-4 rows 4, 5-9 rows 2, 10+ off. The curves show c2/c4/c8 base vs final explicitly (running). (2) The new test is MiniMax's shape: 128-wide heads, 16 query heads a kv head, per-head selection, 1100 keys (every rank walks keys).
- Note 1 (cut, disk), note 2 (mode, config diff, identify bug apart: identify-bug.md), note 3 (numerics: report.md "Tokens and numerics"; cluster off at batching/pbatch: curves running), note 4 (fix 2 measured, not shipped; curves, cross-model, suite once, on the final diff).

## Final-build results since the last check-in
- Curves (results/analysis/curve-table.txt): cut A c2 +1.8%, c4 -0.1%; cut B c2 +2.5%, c4 +1.7%; c8-c64 (cluster off) within +-0.7%; pf4-pf16 within +-0.7% except cut B pf8 -3.1% (a pf fire takes no decode step; 3-round rerun pending).
- Cross-model: Qwen3.8-Flash-Next mini 0.850 -> 0.822 ms (+3.4%); Qwen3-0.6B 1.029 -> 1.028 ms, tokens identical.

## Running / pending
- Curves (results/raw/curve), cross-model on the final engine (x-*-final), GPU suite (results/tests/gpu-suite-*.log), final rebase onto upstream main (only a65475094, vllm_bench) + one A/B round.
