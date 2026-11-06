# Exposed (critical-path) time per kernel class in pie decode steps: on the step's main stream, each kernel
# is charged end_i - end_(i-1); side-stream kernels are charged only where they delay the main stream (not here).
import sqlite3, sys
from collections import defaultdict
c = sqlite3.connect(sys.argv[1])
rows = c.execute("""select k.start,k.end,s.value,k.gridX,k.gridY,k.blockX,k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k
  join StringIds s on s.id=k.shortName order by k.start""").fetchall()
idx = [i for i, r in enumerate(rows) if r[2] == "settle"]
steps = [rows[a + 1:b + 1] for a, b in zip(idx, idx[1:])]
main = steps[0][0][6]
exp = defaultdict(float); n = len(steps); tot = 0
for s in steps:
    m = sorted((r for r in s if r[6] == main), key=lambda r: r[1])
    prev = m[0][0]
    for r in m:
        key = f"{r[2][:40]} {r[3]}x{r[4]}"
        exp[key] += (r[1] - prev) / 1e3 / n; prev = r[1]
    tot += (m[-1][1] - m[0][0]) / 1e3 / n
print(f"main-stream span {tot:.1f} us/step over {n} steps")
for k, v in sorted(exp.items(), key=lambda x: -x[1]):
    if v > 5: print(f"{v:8.1f} us  {k}")
