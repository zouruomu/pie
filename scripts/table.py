# Before/after table from bench JSONs: median output tok/s per engine tag and shape, with min-max and n.
# usage: table.py <dir> <shapes, comma-separated> <tag>...   (files are <tag>-<shape>-r<N>.json)
import glob
import json
import os
import statistics
import sys

d, shapes, tags = sys.argv[1], sys.argv[2].split(","), sys.argv[3:]
vals = {}
for t in tags:
    for s in shapes:
        v = [json.load(open(f))["summary"]["output_tok_per_s"] for f in glob.glob(os.path.join(d, f"{t}-{s}-r*.json"))]
        vals[t, s] = v
print("| tok/s | " + " | ".join(shapes) + " |")
print("|---|" + "---|" * len(shapes))
for t in tags:
    cells = []
    for s in shapes:
        v = vals[t, s]
        cells.append(f"{statistics.median(v):.1f} ({min(v):.1f}-{max(v):.1f}, n={len(v)})" if v else "-")
    print(f"| {t} | " + " | ".join(cells) + " |")
if len(tags) >= 2:
    base, fix = tags[0], tags[1]
    print("| fix / base | " + " | ".join(
        f"{statistics.median(vals[fix, s]) / statistics.median(vals[base, s]) - 1:+.1%}" if vals[fix, s] and vals[base, s] else "-"
        for s in shapes) + " |")
    if len(tags) >= 3:
        ref = tags[2]
        for t in (base, fix):
            print(f"| {t} / {ref} | " + " | ".join(
                f"{statistics.median(vals[t, s]) / statistics.median(vals[ref, s]) - 1:+.1%}" if vals[t, s] and vals[ref, s] else "-"
                for s in shapes) + " |")
