# Curve table across rounds: for each engine group and shape, the metric's mean, min-max and spread.
# usage: curve.py name=glob_prefix[,glob_prefix...] ...   (a run file is <prefix>-<shape>-r<N>.json)
# metric: pf* -> prefill tok/s (sum prompt tokens / wall); c* -> output tok/s; raw -> decode tok/s
# (per-request (n-1)/(latency-ttft), median) and TTFT p50.
import glob, json, statistics as st, sys, os
SHAPES = "pf4 pf8 pf16 pf32 pf64 pf128 pf256 pf512 pf1024 c4 c8 c16 c32 c64 c128 c256 c512 c1024 raw".split()
groups = [a.split("=", 1) for a in sys.argv[1:]]

def metric(d, shape):
    s = d["summary"]
    if shape.startswith("pf"):
        return s["prompt_tokens"] / s["wall_s"]
    if shape.startswith("c"):
        return s["output_tok_per_s"]
    lanes = [(r["output_tokens"] - 1) / (r["latency_s"] - r["ttft_s"]) for r in d["requests"]
             if r.get("ok") and r.get("ttft_s") is not None and r["latency_s"] > r["ttft_s"]]
    if not lanes:  # latency mode without per-request ttft (vLLM): output tokens / latency
        lanes = [r["output_tokens"] / r["latency_s"] for r in d["requests"] if r.get("ok")]
    return st.median(lanes)

table = {}
for name, prefixes in groups:
    for shape in SHAPES:
        vals = []
        for p in prefixes.split(","):
            for f in sorted(glob.glob(f"{p}-{shape}-r*.json")):
                d = json.load(open(f))
                if d["summary"]["completed"] == d["summary"]["requests"]:
                    vals.append(metric(d, shape))
        table[(name, shape)] = vals
names = [g[0] for g in groups]
print("| shape | " + " | ".join(f"{n} mean (min-max, n)" for n in names) + " |")
print("|---|" + "---|" * len(names))
for shape in SHAPES:
    cells = []
    for n in names:
        v = table[(n, shape)]
        cells.append(f"{st.mean(v):.1f} ({min(v):.0f}-{max(v):.0f}, {len(v)})" if v else "-")
    print(f"| {shape} | " + " | ".join(cells) + " |")
for n in names:
    best = max(((st.mean(v), s) for (m, s), v in table.items() if m == n and v and s != "raw" and s.startswith("c")), default=None)
    bestp = max(((st.mean(v), s) for (m, s), v in table.items() if m == n and v and s.startswith("pf")), default=None)
    print(f"{n}: best batching {best[0]:.1f} at {best[1]}; best pbatch {bestp[0]:.1f} at {bestp[1]}" if best and bestp else n)
