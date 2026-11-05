# Summarise bench JSONs: label, completed, wall, output tok/s, TTFT p50, latency p50, ITL p50,
# lanes decoding at once (output tok/s x ITL p50), SM clock at end, engine sha256 (pie only).
# usage: summ.py <glob>...   (prints one row per JSON, then a per-label-prefix median table)
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict

rows = []
for f in sorted(sum((glob.glob(a) for a in sys.argv[1:]), [])):
    if not f.endswith(".json"):
        continue
    try:
        d = json.load(open(f))
    except Exception as e:  # noqa: BLE001
        print(os.path.basename(f), "BAD", e)
        continue
    s = d["summary"]
    c = s.get("config") or {}
    clk = c.get("gpu clocks at end") or [{}]
    clk = clk[0] if isinstance(clk, list) and clk else (clk or {})
    out = s["output_tok_per_s"]
    itl = s.get("intertoken_p50_ms") or 0
    label = os.path.basename(f)[:-5]
    sha = (c.get("embedded engine sha256") or "")[:8]
    rows.append((label, out))
    print(
        f"{label:40s} {s['engine']:5s} ok={s['completed']}/{s['requests']} wall={s['wall_s']:7.1f} "
        f"out={out:8.1f} ttft50={(s.get('ttft_p50_ms') or 0) / 1000:7.1f}s "
        f"lat50={(s.get('latency_p50_ms') or 0) / 1000:7.1f}s itl50={itl:6.2f} "
        f"lanes={out * itl / 1000:5.1f} sm={clk.get('sm_mhz', '')} sha={sha}"
    )

groups = defaultdict(list)
for label, out in rows:
    groups[re.sub(r"-r\d+$", "", label)].append(out)
print()
print(f"{'point':40s} {'n':>2s} {'median':>8s} {'min':>8s} {'max':>8s}")
for key in sorted(groups):
    v = groups[key]
    print(f"{key:40s} {len(v):2d} {statistics.median(v):8.1f} {min(v):8.1f} {max(v):8.1f}")
