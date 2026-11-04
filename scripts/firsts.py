# For each graph id: when it first runs, how often, and the device-idle gap right before its first run.
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
g = c.execute("select start,end,graphId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE where start >= 0 order by start").fetchall()
k = c.execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL where graphId is null or graphId=0").fetchall()
ev = sorted([(s, e, gid) for s, e, gid in g] + [(s, e, None) for s, e in k])
t0 = ev[0][0]; ce = ev[0][1]; first = {}; count = {}; gapsum_first = 0
for s, e, gid in ev[1:]:
    gap = max(0, s - ce)
    if gid is not None:
        count[gid] = count.get(gid, 0) + 1
        if gid not in first:
            first[gid] = (s, gap, e - s); gapsum_first += gap
    ce = max(ce, e)
for gid, (s, gap, d) in sorted(first.items(), key=lambda kv: kv[1][0]):
    print(f"g{gid:<5} first at {(s-t0)/1e9:7.2f}s  gap before {gap/1e6:8.1f} ms  dur {d/1e6:7.1f} ms  runs {count[gid]}")
print(f"idle before first runs: {gapsum_first/1e6:.1f} ms over {len(first)} graphs")
