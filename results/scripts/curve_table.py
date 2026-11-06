# Base vs fix curve table from results/raw/curve/<cut>-<engine>-<point>.json: output tok/s for the
# batching points, prefill tok/s for the pf points, and ITL p50 where the client reports it.
import json, sys
from pathlib import Path
R = Path("/root/r2g/evidence/results/raw/curve")
points = ["c4", "c8", "c16", "c32", "c64", "pf4", "pf8", "pf16", "pf32"]
def get(cut, eng, pt):
    f = R / f"{cut}-{eng}-{pt}.json"
    if not f.exists():
        return None
    s = json.load(open(f))["summary"]
    if pt.startswith("pf"):
        n = s.get("prompt_tokens") or 0
        return n / s["wall_s"] if s.get("wall_s") else None, None
    return s.get("output_tok_per_s"), s.get("intertoken_p50_ms")
for cut in ("a", "b"):
    print(f"cut {cut}: point | base | fix | fix/base")
    for pt in points:
        b, f = get(cut, "base", pt), get(cut, "fix", pt)
        if b is None or f is None or not b[0] or not f[0]:
            print(f"  {pt}: pending"); continue
        itl = f" (ITL p50 {b[1]:.2f} -> {f[1]:.2f} ms)" if b[1] and f[1] else ""
        print(f"  {pt}: {b[0]:.1f} | {f[0]:.1f} | {f[0]/b[0]:.3f}{itl}")
