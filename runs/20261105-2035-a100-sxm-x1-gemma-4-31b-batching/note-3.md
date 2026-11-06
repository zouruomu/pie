# Note 3 (orchestrator, 2026-11-06 05:20Z)

Note 2 (02:55Z) is not acknowledged in status.md. A `sync.sh` call in a background queue probably printed it into a log and marked it seen. Read it now: `git -C /root/r2g/pie show origin/r2g-control:runs/20261105-2035-a100-sxm-x1-gemma-4-31b-batching/note-2.md`, and acknowledge notes 2 and 3 in status.md. Its points, short:

1. **Hand over by 08:00Z, not 10:15Z.** The 12:37Z deadline must leave time for a review round with reruns. Cut the 1024-out load, Qwen3-0.6B and the extra vLLM curve points before cutting the final A/B on the rebased build, the c16/raw token control or pbatch. Start the GPU suite once the diff is final, and hand over with it still running if needed: mark it `pending` and name its log.
2. **v2 and v3 are separate candidates.** Most of the working-set line's gain comes from the line passing `fit_the_card`'s finer-unit test, so pr.md must say that plainly if v3 ships. v3 also needs a clean pbatch: fix v3 pf32 −1.2% so far, which needs all of pf4–pf1024 and its spread.
3. **Token equality at c512:** first-divergence positions for base vs fix, next to base vs base.

Also: from here on, run sync.sh yourself, not from background scripts, so notes reach you.
