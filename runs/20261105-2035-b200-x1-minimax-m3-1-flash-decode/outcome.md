# Outcome: 20261105-2035-b200-x1-minimax-m3-1-flash-decode

**Abandoned at 52 min, with no measurement.**

- **Pod:** `wn5d3st1b6mb9t`, 1 × B200, EU-RO-1, $7.99/h. Created 20:46:23Z on the first retry after "no instances available".
- **Agent:** started 20:47:38Z, exited with code 143 (SIGTERM) at 20:54:09Z, 6½ minutes in. `agent-1.json` is empty and `agent-1.err` stayed on the pod, so the cause is unknown. The bootstrap then idled, waiting for a review, as designed.
- **The stop:** a person stopped the pod ("Exited by user") at 21:38:05Z. The parallel H100 run's pod was stopped at the same second, just after teammates resumed mi300x-eval-1..4. The orchestrator didn't stop it, and the bootstrap deletes rather than stops.
  - The pod was not restarted over a person's action. The orchestrator deleted the stopped pod at 21:49Z; a stop had already wiped its container disk.
- **Cost:** about $7.

## What the evidence holds

Only the agent's first status.md, written about 20:54Z, which says 21:05 (a wrong clock line):
- It cut MiniMaxAI/MiniMax-M3.1-Flash @ce7f4df8 with `scripts/bench/shrink_checkpoint.py`: layers 0–59, 16 of 128 experts, 43.4 GiB (MXFP8 projections, NVFP4 experts).
- anton-mel/stats has no MiniMax recipe, and pie-project/benchmarks isn't readable to the pod. So the cut follows the site's description, not the harness itself.
- The site gives this miniature 124.25 B parameters, which a 16-expert cut doesn't obviously match. Note 1 asked about it; the agent never read it.

## For a retry

- The cell is still fresh: nothing that touched `profiles/sm100*`, MiniMax or sm_100 FP8 landed on main through e4bb499.
- The brief (`runs/<this run>/brief.md`) can be reused as is, with a new run id.
- Keep an eye on the first 10 minutes. This agent was killed with SIGTERM before its first real check-in. If it happens again, have the bootstrap keep `agent-N.err` in the evidence (an infrastructure change, so Felix's call).
