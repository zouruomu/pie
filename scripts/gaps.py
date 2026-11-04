# Device-idle gaps between consecutive GPU work (graphs + eager kernels + memcpy) in an nsys export:
# histogram of gap sizes, and the largest gaps with the graphs on either side.
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
ev = [(s, e, f"g{gid}") for s, e, gid in c.execute("select start,end,graphId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE where start >= 0")]
ev += [(s, e, "k:" + n) for s, e, n in c.execute("select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.graphId is null or k.graphId=0")]
ev += [(s, e, f"memcpy{kd}") for s, e, kd in c.execute("select start,end,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY")]
ev.sort()
gaps = []; ce = ev[0][1]; prev = ev[0][2]
for s, e, n in ev[1:]:
    if s > ce: gaps.append(((s - ce) / 1e6, prev, n, s))
    if e > ce: ce = e; prev = n
bins = [0.05, 0.2, 1, 5, 20, 100, 1e9]
lo = 0
for b in bins:
    sel = [g for g in gaps if lo <= g[0] < b]
    print(f"gaps {lo:>6}-{b:<6} ms: n={len(sel):6d} total {sum(g[0] for g in sel):9.1f} ms")
    lo = b
t0 = ev[0][0]
for g in sorted(gaps, reverse=True)[:12]:
    print(f"  {g[0]:8.1f} ms at t={(g[3]-t0)/1e9:7.2f}s  after {g[1][:40]}  before {g[2][:40]}")
