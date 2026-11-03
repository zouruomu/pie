# Merge graph executions (graph-level trace) and stand-alone kernels into one
# GPU timeline; print a window and per-step stats.
import sqlite3, sys, statistics as st
db = sys.argv[1]; start_i = int(sys.argv[2]) if len(sys.argv) > 2 else None; n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
c = sqlite3.connect(db)
ev = [(s, e, "GRAPH g%d" % g, sid) for s, e, g, sid in c.execute("select start,end,graphId,streamId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE")]
ev += [(s, e, nm, sid) for s, e, nm, sid in c.execute("select k.start,k.end,s.value,k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName")]
try:
    ev += [(s, e, "MEMCPY%d %dB" % (k, b), sid) for s, e, k, b, sid in c.execute("select start,end,copyKind,bytes,streamId from CUPTI_ACTIVITY_KIND_MEMCPY")]
    ev += [(s, e, "MEMSET %dB" % b, sid) for s, e, b, sid in c.execute("select start,end,bytes,streamId from CUPTI_ACTIVITY_KIND_MEMSET")]
except Exception as ex:
    print("no memcpy", ex)
ev.sort()
from collections import Counter
big = [x for x in ev if x[2].startswith("GRAPH") and x[1] - x[0] > 1_500_000]
print("events", len(ev), "graphs>1.5ms", len(big), Counter(x[2] for x in big))
d = [ (x[1]-x[0])/1000 for x in big]
if d: print("big graph dur median %.1f min %.1f max %.1f" % (st.median(d), min(d), max(d)))
p = [(b[0]-a[0])/1000 for a, b in zip(big, big[1:]) if (b[0]-a[0]) < 3_000_000]
if p: print("big graph start-to-start median %.1f (n=%d)" % (st.median(p), len(p)))
gaps = []
for a, b in zip(big, big[1:]):
    if b[0] - a[0] > 3_000_000: continue
    between = [x for x in ev if a[1] <= x[0] < b[0]]
    gaps.append(((b[0]-a[1])/1000, sum((x[1]-x[0]) for x in between)/1000, len(between)))
if gaps:
    print("end-of-graph to next-graph: median %.1f us; work in between median %.1f us over median %d events" % (
        st.median(g[0] for g in gaps), st.median(g[1] for g in gaps), st.median(g[2] for g in gaps)))
if start_i is not None:
    t0 = ev[start_i][0]; pe = None
    for x in ev[start_i:start_i+n]:
        print("t=%9.2f dur=%8.2f gap=%7.2f s%s %s" % ((x[0]-t0)/1000, (x[1]-x[0])/1000, (x[0]-pe)/1000 if pe else 0, x[3], x[2][:90]))
        pe = x[1] if pe is None else max(pe, x[1])
