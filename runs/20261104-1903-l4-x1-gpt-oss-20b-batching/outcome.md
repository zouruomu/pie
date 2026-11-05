# Outcome: 20261104-1903-l4-x1-gpt-oss-20b-batching

**Verdict:** shipped as draft PR pie-project/pie#1432 (head `8e8b19b0` on upstream `48399500`, the newest main), opened from Felix's interactive session. This was a follow-up to `20261103-2218-l4-x1-gpt-oss-20b-batching`: it finished that run's side fix and served as the canary for infra `4c38bd5`.

## The change

- **Profile:** in the L4 profile, `moe.mxfp4_mma_per_from` goes from 4 to 0. gpt-oss-20b's c7–c24 decode selects (1–3 routes an expert) move to the 512-code run kernel.
- **Test:** one kernel-test case, gpt-oss-20b's c16 gate/up select, `(K::MxBias, mma, [32, 16, 4, 320, 2880], t, alpha, 0)`. It fires `moe_select_mxfp4_mma_run<4, 4, 3, 4>` and passes.

## Numbers (orchestrator recomputed from the raw JSONs; every file carries its build's sha256)

| shape | base, 3 rounds on `1b851247` | fix | change | base, 3 rounds rebased on `48399500` | fix | change | vLLM 0.32.0 |
|---|---|---|---|---|---|---|---|
| c8 | 242.9 | 277.4 | +14.2% | 238.6 | 277.1 | +16.1% | 275.6 |
| c16 | 301.0 | 485.9 | +61.4% | 297.7 | 491.2 | +65.0% | 441.9 |
| c24 | 673.6 | 664.5 | −1.3% (below base in each round) | 663.5 | 656.3 | −1.1% | – |
| c32 (control) | 835.7 | 839.0 | +0.4% | 818.0 | 807.4 | −1.3% (base spans 798–852) | 737.0 |

- **Clocks:** the 72 W card holds a higher clock on the fix at c8 (about 1,500 against 1,030 MHz).
- **Tokens:** base against fix sits at the low end of the reruns' range at c8/c16 (every step changes kernel) and inside it at c32.

## Tests (read from the logs)

- **kernels-cuda `moe_select`:** 5 passed on both builds. `the_routed_selects_answer_the_routed_dot` fails identically on both.
  - A fire trace shows every case before the failure passing, the new case included.
  - The failure is #1274's first W4A8 case. Off sm_120a/sm_121a, `select_mxfp4_w4a8` returns `Ok(false)`, the test accepts that, and it then checks the untouched 0x7f7f fill.
  - It is a pre-existing test bug, reported to Anton. It corrects run 1's attribution to the pipe kernel; the pipe kernel passes on sm_89.
- **compute-sanitizer:** no launch or CUDA errors.
- **engine-cuda GPU suite** (fix tree, single-threaded, dev build at opt-level 2): 55 passed, 4 failed, 1 ignored, in 4,412 s.
  - All 4 failures were device-memory starvation beside the pod's own parallel reruns.
  - Alone on the GPU, all 4 pass on fix (`idle-joined-fix.log`, `idle-synth-fix.log`) and on base (`joined-base2.log`, `idle-synth-base.log`).
  - glm45-air-mini's "joined model picks the same token" failed once on each build while memory was short: a near-tie.
- **Rebase:** #1429 only touches `models/glm_5_next.poem`, and the rebased commit is patch-identical, so the suites were not rerun.

## Review

- **Fixes made by the orchestrator in the PR text:**
  - "under the 2-an-expert grouping threshold" became "not past the grouped select's 2 an expert" (c16 is exactly 2);
  - "off sm_120a" became "off sm_120a and sm_121a";
  - the tests now have their own paragraph.
- **A status slip:** one status line had claimed `joined-fix2.log` passed. It failed (OOM on hy4-mini), and the later idle rerun passed. The final status and pr.md were right.
- **A small inconsistency, left as is:** the profile comment cites run 1's alternative-table numbers (c16 297 to 485, c8 241 to 280). They are real, but differ slightly from the PR's.

## Infra canary (4c38bd5)

- **Secrets:** bootstrap had all three tokens set. The agent's tool shell had only `CLAUDE_CODE_MESSAGING_TOKEN`, which the pod stripped from every build and bench.
- **Sync:** the HF download worked, and sync pushed through the credential file.
- **Notes:** note-1 was received at the first sync and acknowledged.
- **Status:** hourly updates held until the last stretch, each with an ETA.
- **Hand-over** came at 01:39Z, 6.6 h after the pod was created; the target was about 5 h. The engine-cuda suite in a debug build was too slow, and the memory-starved reruns cost about 2 h.
- **Idle time:** the orchestrator (an interactive session's local poll, which required a `**Phase:**` line) missed the READY marker. The pod idled from 01:39Z until its deadline at 03:16Z, then terminated itself (GET 404). No review rounds were needed.

## Cost

- **Pod 5kbv49df0rekbk:** NVIDIA L4, $0.59/h, EUR-IS-1.
- **Time and total:** 2026-11-04 19:04Z to 2026-11-05 03:16Z, 8.2 h, about $4.85. About $0.95 of that was the idle stretch after hand-over.
