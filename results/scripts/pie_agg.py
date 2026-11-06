# Per-step kernel time of a pie nsys node-trace sqlite: steps end at `settle`; groups by short name + grid.
import sqlite3, sys
from collections import defaultdict
c = sqlite3.connect(sys.argv[1])
rows = c.execute("""select k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ,k.blockX from CUPTI_ACTIVITY_KIND_KERNEL k
  join StringIds s on s.id=k.shortName order by k.start""").fetchall()
idx = [i for i, r in enumerate(rows) if r[2] == "settle"]
steps = [rows[a + 1:b + 1] for a, b in zip(idx, idx[1:])]
n = len(steps)
period = sum(s[-1][1] - s[0][0] for s in steps) / n / 1e3
busy = defaultdict(float); cnt = defaultdict(int)
for s in steps:
    for r in s:
        key = (r[2][:60], f"{r[3]}x{r[4]}x{r[5]}", r[6])
        busy[key] += (r[1] - r[0]) / 1e3 / n; cnt[key] += 1 / n
# union of busy intervals per step (overlap-aware)
union = 0
for s in steps:
    iv = sorted((r[0], r[1]) for r in s); cur_s, cur_e = iv[0]; u = 0
    for a, b in iv[1:]:
        if a > cur_e: u += cur_e - cur_s; cur_s, cur_e = a, b
        else: cur_e = max(cur_e, b)
    u += cur_e - cur_s; union += u / 1e3 / n
print(f"steps {n}, step period {period:.1f} us, kernels/step {len(steps[0])}, sum kernel {sum(busy.values()):.1f} us, busy union {union:.1f} us, idle {period-union:.1f} us")
for k, v in sorted(busy.items(), key=lambda x: -x[1])[:int(sys.argv[2]) if len(sys.argv) > 2 else 40]:
    print(f"{v:8.1f} us {cnt[k]:6.1f}x {v/cnt[k]:7.2f} us/each  {k[0]} grid {k[1]} blk {k[2]}")
