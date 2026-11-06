# Per-step kernel time of a vLLM nsys node-trace sqlite: a step starts at `_prepare_pos_seq_lens_kernel`.
# Prints period, busy union, and per-kernel sum and exposed time (end_i - max(end_(i-1)) over all streams).
import sqlite3, sys
from collections import defaultdict
c = sqlite3.connect(sys.argv[1])
rows = c.execute("""select k.start,k.end,s.value,k.gridX,k.gridY,k.blockX,k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k
  join StringIds s on s.id=k.shortName order by k.start""").fetchall()
mark = sys.argv[2] if len(sys.argv) > 2 else "_prepare_pos_seq_lens_kernel"
idx = [i for i, r in enumerate(rows) if r[2] == mark]
steps = [rows[a:b] for a, b in zip(idx, idx[1:])]
# decode steps only: drop the outliers (prefill) by kernel count
from statistics import median
mc = median(len(s) for s in steps)
steps = [s for s in steps if len(s) == mc]
n = len(steps)
period = sum(b[0][0] - a[0][0] for a, b in zip(steps, steps[1:]) if True) / (n - 1) / 1e3
busy = defaultdict(float); exp = defaultdict(float); cnt = defaultdict(float); union = 0
for s in steps:
    s = sorted(s, key=lambda r: r[1]); last = s[0][0]
    for r in sorted(s, key=lambda r: r[0]):
        pass
    for r in s:
        key = f"{r[2][:60]} {r[3]}x{r[4]}"
        busy[key] += (r[1] - r[0]) / 1e3 / n; cnt[key] += 1 / n
        exp[key] += max(0, r[1] - max(last, r[0])) / 1e3 / n; last = max(last, r[1])
    union += sum(max(0, 0) for _ in [0])
tot_exp = sum(exp.values())
print(f"decode steps {n} ({mc} kernels each), step period {period:.1f} us, exposed kernel time {tot_exp:.1f} us, gaps {period - tot_exp:.1f} us")
for k, v in sorted(exp.items(), key=lambda x: -x[1])[:int(sys.argv[3]) if len(sys.argv) > 3 else 45]:
    print(f"{v:8.1f} us exposed {busy[k]:8.1f} us sum {cnt[k]:5.1f}x  {k}")
