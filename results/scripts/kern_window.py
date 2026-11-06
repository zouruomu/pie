# Kernel totals by name inside an NVTX range (any process of the trace), as a CSV kern_table.py reads.
# usage: kern_window.py <trace.sqlite> <nvtx range text> <out.csv>
import csv, sqlite3, sys
from collections import defaultdict

db, text, out = sys.argv[1:4]
c = sqlite3.connect(db)
r = c.execute("select start, end from NVTX_EVENTS where text = ? order by start desc limit 1", (text,)).fetchone()
if r is None:
    r = c.execute("select n.start, n.end from NVTX_EVENTS n join StringIds s on s.id = n.textId where s.value = ? "
                  "order by n.start desc limit 1", (text,)).fetchone()
t0, t1 = r
tot, cnt = defaultdict(int), defaultdict(int)
for name, a, b in c.execute("select s.value, k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k "
                            "join StringIds s on s.id = k.demangledName where k.start >= ? and k.start < ?", (t0, t1)):
    tot[name] += b - a
    cnt[name] += 1
w = csv.writer(open(out, "w"))
w.writerow(["Total Time (ns)", "Instances", "Name"])
for n in sorted(tot, key=lambda n: -tot[n]):
    w.writerow([tot[n], cnt[n], n])
busy = sum(tot.values())
print(f"window {(t1 - t0) / 1e6:.1f} ms, kernel time {busy / 1e6:.1f} ms, {sum(cnt.values())} kernels")
