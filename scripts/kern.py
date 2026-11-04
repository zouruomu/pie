# Kernel time by name in an nsys sqlite export: total ms, calls, share, over the captured window.
import sqlite3, sys
from collections import defaultdict
c = sqlite3.connect(sys.argv[1])
rows = c.execute("select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName").fetchall()
t0 = min(r[0] for r in rows); t1 = max(r[1] for r in rows)
agg = defaultdict(lambda: [0, 0])
for s, e, n in rows:
    agg[n][0] += e - s; agg[n][1] += 1
tot = sum(v[0] for v in agg.values())
print(f"window {(t1-t0)/1e6:.1f} ms, kernel time {tot/1e6:.1f} ms, {len(rows)} kernels")
for n, (d, k) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:int(sys.argv[2]) if len(sys.argv) > 2 else 25]:
    print(f"{d/1e6:9.2f} ms {k:7d} {100*d/tot:5.1f}%  {n[:90]}")
