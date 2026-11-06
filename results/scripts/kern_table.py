# Group an nsys cuda_gpu_kern_sum CSV into components; with two CSVs, print them side by side.
# usage: kern_table.py label=a.csv [label=b.csv ...] [--top N]
import csv, re, sys
from collections import defaultdict

def component(name):
    l = name.lower()
    # MoE expert GEMMs: pie's MXFP4 selects; TRT-LLM's CUTLASS grouped (GroupProblemShape) GEMMs
    if "moe_select_mxfp4" in l or "groupproblemshape" in l:
        return "MoE expert GEMMs (gate-up+down)"
    # MoE glue: routing, permutes, activation quantize, SwiGLU (where not fused), combine, bias
    if any(x in l for x in ("moe_weighted_sum", "moe_bias_sum", "moe_route", "moe_topk", "routing", "expandinputrows",
                            "doactivation", "finalize", "computestrides", "blockexpertprefix", "quantize_with_block_size",
                            "mergeexpertprefix", "unpermute", "topk")):
        return "MoE routing/quantize/SwiGLU/combine"
    if "fmha" in l or "batchprefillwithpagedkvcache" in l or "attention" in l and "rmsnorm" not in l:
        return "attention"
    if any(x in l for x in ("nvjet", "cutlass_80_", "gemm", "gemv", "cublas", "splitkreduce")):
        return "dense GEMMs (qkv, o, router, head)"
    if any(x in l for x in ("rmsnorm", "row_skeleton", "rope", "kvcache", "norm")):
        return "norms, residuals, rope, KV write"
    return "other"


def load(path):
    out = defaultdict(float)
    names = defaultdict(float)
    for r in csv.DictReader(open(path)):
        t = float(r["Total Time (ns)"]) / 1e6
        out[component(r["Name"])] += t
        names[r["Name"]] += t
    return out, names


args = [a for a in sys.argv[1:] if "=" in a]
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 12
tables = [(a.split("=", 1)[0], *load(a.split("=", 1)[1])) for a in args]
comps = sorted({c for _, t, _ in tables for c in t}, key=lambda c: -max(t.get(c, 0) for _, t, _ in tables))
print("| component | " + " | ".join(f"{l} ms" for l, _, _ in tables) + (" | delta ms |" if len(tables) == 2 else " |"))
print("|---" * (len(tables) + 1 + (len(tables) == 2)) + "|")
for c in comps:
    v = [t.get(c, 0) for _, t, _ in tables]
    print(f"| {c} | " + " | ".join(f"{x:.1f}" for x in v) + (f" | {v[0]-v[1]:+.1f} |" if len(v) == 2 else " |"))
tot = [sum(t.values()) for _, t, _ in tables]
print("| **total kernel time** | " + " | ".join(f"**{x:.1f}**" for x in tot) + (f" | **{tot[0]-tot[1]:+.1f}** |" if len(tot) == 2 else " |"))
for l, _, names in tables:
    print(f"\n{l}: top kernels")
    for n, t in sorted(names.items(), key=lambda x: -x[1])[:top]:
        print(f"  {t:9.1f} ms  {component(n):32s} {n[:150]}")
