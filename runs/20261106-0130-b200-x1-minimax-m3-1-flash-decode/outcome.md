# Outcome: 20261106-0130-b200-x1-minimax-m3-1-flash-decode

**Shipped: pie-project/pie#1485 (draft).** Head `d06613bc4` on `r2g/20261106-0130-b200-x1-minimax-m3-1-flash-decode`, one commit, rebased on main dc6282d7.

(The run id says 0130; the claim was actually made at 01:10Z.)

## Verdict
- **The site's cell can't be reproduced as stated.** No released vLLM (0.32.0, 0.33.0) or SGLang (0.5.22) loads MiniMax-M3.1-Flash: they read its NVFP4 expert scales as MXFP8. So the site's 174.43 / 128.24 tok/s didn't come from this checkpoint at those versions.
- **The gap is real on the same weights.** The same 60-layer, 16-expert cut of MiniMax-M3-MXFP8 (cut B) loads everywhere. On it, pie trails vLLM 0.32.0 by 28.0% and SGLang 0.5.22 by about 22% at c1.
- **Fix:** the decode's selected attention, which ran on one block per kv head, now runs over a cluster of up to 8 blocks per kv head, folded through distributed shared memory. It is on in the B200 profile only.
- **Numbers I recomputed from the raw files:**
  - M3.1-Flash: 8.976 → 8.620 ms ITL p50 (+4.1%).
  - M3-MXFP8: 9.623 → 9.286 ms (+3.6%). Against vLLM's 6.925 ms, the gap goes from 28.0% to 25.4%.
  - c2/c4 gain slightly; c8–c64 and pf4–pf32 are within spread (the pf8 reruns confirm it).
  - Engine sha256 per run: base b0ca94bb, final fff664e3.
  - Tokens: cut B identical 8/8 × 256. Cut A first diverges at [247, 8, 1, 1, 1, 1, 1, none].
- **Taken on trust:**
  - The KL values: the raw logit dumps weren't committed. logit-cmp.txt and the probe logs are consistent with them.
  - The nsys per-step attribution.
  - That vLLM splits each head 16 ways.
- **Numerics:** after review 1, the reference is a decode-only scalar walk with prefill untouched. The fix's KL is not below it: it is below on 2 of 8 requests and up to 2.5× above on the others, the same order of magnitude, with equal or better top-1 agreement. The PR says so plainly.
- **Tests:** GPU suites (engine-cuda 69/69, 58/59, 5/6; kernels-cuda 34/35, 142/144). Each of the five failures has a matching base log on e2e9d1e.

## Review
- Notes 1–5 were sent during the run.
- Review 1 asked for a like-for-like numerics reference and a corrected token count. Both were done in one round.
- I edited no wording myself.

## Handed over, not shipped
- **q/k norm + RoPE + KV append (~0.7 ms a step):** it runs as two one-CTA epilogue slices of the W8A8 qkv GEMM. A reverted prototype of the retired hand rule bought about 4%. The shippable form is skeleton/compiler work.
- **Expert glue (~0.6 ms):** cut B's MXFP8 experts take 4 launches between the expert GEMMs.
- **Cut A's W4A4 gate-up:** it runs at about 2.3 TB/s.

## Side findings
- pie identifies both cuts as the mini's bf16 deployment and OOMs widening them (`identify-bug.md` on the evidence branch).
- pie reads M3.1-Flash's first three layers as dense attention, though its config indexes all 60. The two read the same keys at 768 tokens and differ past 2048.
- The site's 124.25 B parameter count doesn't follow from a 16-expert cut, which comes to 64.67 B.

## Time and cost
- Pod created 01:13Z (B200, US-CA-2, $7.99/h).
- Early verdict 02:57Z (1 h 44 min in).
- Hand-over 08:35Z (7 h 22 min). Review 1 answered 08:53Z.
- Closed 09:38Z, against a 21:13Z deadline.
- About 8.5 h of pod time, roughly $68.
