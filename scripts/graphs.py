# Graph-level nsys export: per-graph-id count / mean / total, device busy vs idle over the window.
# Only records at t >= 0 (the capture start): the export also holds graph executions from load-time arming, before cudaProfilerStart.
import sqlite3, sys, statistics as st
from collections import defaultdict
c = sqlite3.connect(sys.argv[1])
g = c.execute("select start, end, graphId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE where start >= 0 order by start").fetchall()
k = c.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL where graphId is null or graphId = 0").fetchall()
ev = sorted([(s, e) for s, e, _ in g] + k)
t0, t1 = ev[0][0], max(e for _, e in ev)
busy = 0; cs, ce = ev[0]
for s, e in ev[1:]:
    if s > ce: busy += ce - cs; cs, ce = s, e
    else: ce = max(ce, e)
busy += ce - cs
print(f"window {(t1-t0)/1e6:.1f} ms  busy {busy/1e6:.1f} ms  idle {(t1-t0-busy)/1e6:.1f} ms  graphs {len(g)}  eager kernels {len(k)} ({sum(e-s for s,e in k)/1e6:.1f} ms)")
by = defaultdict(list)
for s, e, gid in g: by[gid].append((e - s) / 1e6)
rows = sorted(by.items(), key=lambda kv: -sum(kv[1]))
for gid, d in rows[:int(sys.argv[2]) if len(sys.argv) > 2 else 20]:
    print(f"graph {gid:6d} n={len(d):5d} mean {st.mean(d):8.2f} ms  median {st.median(d):8.2f}  total {sum(d):9.1f} ms")
