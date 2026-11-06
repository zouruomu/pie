# Per-step component table from an nsys node-trace sqlite (pie or vLLM decode). Each kernel is charged its
# exposed time: end - max(previous ends on any stream, its start); what is left of the step period is "gaps".
# usage: components.py pie|vllm trace.sqlite
import re, sqlite3, sys
from collections import defaultdict
from statistics import median
eng, db = sys.argv[1], sys.argv[2]
c = sqlite3.connect(db)
rows = c.execute("""select k.start,k.end,s.value,d.value,k.gridX,k.gridY from CUPTI_ACTIVITY_KIND_KERNEL k
  join StringIds s on s.id=k.shortName join StringIds d on d.id=k.demangledName order by k.start""").fetchall()
if eng == "pie":
    idx = [i + 1 for i, r in enumerate(rows) if r[2] == "settle"]
else:
    idx = [i for i, r in enumerate(rows) if r[2] == "_prepare_pos_seq_lens_kernel"]
steps = [rows[a:b] for a, b in zip(idx, idx[1:])]
mc = median(len(s) for s in steps)
pairs = [(s, t) for s, t in zip(steps, steps[1:]) if len(s) == mc]
RULES = {
    "pie": [
        ("routed experts GEMMs", lambda n, d, g: n == "tc5_gemm" and g == "148x1"),
        ("projections (qkv, index q/k, o, shared expert, dense MLP)", lambda n, d, g: n == "tc5_gemm"),
        ("activation quantize (e4m3 1x32)", lambda n, d, g: n.startswith("act_quant")),
        ("routing: gate, top-k, layout, gather, weighted sum", lambda n, d, g: n.startswith(("moe_gate", "route_", "moe_weighted"))),
        ("SwiGLU", lambda n, d, g: n.startswith("swiglu")),
        ("norms, q/k norm + rope, kv append", lambda n, d, g: n in ("row_skeleton", "rmsnorm_in_warp", "rope_partial", "mla_kv_append")),
        ("index top-k", lambda n, d, g: n.startswith("index_topk")),
        ("attention (selected + 3 dense layers)", lambda n, d, g: n.startswith(("sdpa_", "BatchDecode", "PersistentVariable"))),
        ("lm_head", lambda n, d, g: n.startswith("nvjet")),
    ],
    "vllm": [
        ("routed experts GEMMs", lambda n, d, g: n.startswith("bmm_")),
        ("projections (qkv, index q/k, o, shared expert, dense MLP)", lambda n, d, g: "dense_blockscaled_gemm" in d),
        ("activation quantize (e4m3 1x32)", lambda n, d, g: "mxfp8_quantize" in d),
        ("routing: gate, top-k, layout, gather, weighted sum", lambda n, d, g: "router_gemm" in d or n in ("_splitk_reduce_kernel", "routingIndicesBlockKernel", "finalizeKernel")),
        ("SwiGLU", lambda n, d, g: n == "act_and_mul_kernel"),
        ("norms, q/k norm + rope, kv append", lambda n, d, g: "rmsnorm" in d.lower() or n.startswith("fusedMiniMaxM3") or n in ("vectorized_elementwise_kernel", "reshape_and_cache_flash_kernel")),
        ("index top-k", lambda n, d, g: "index_decode_score" in d or n.startswith("IndexerTopK")),
        ("attention (selected + 3 dense layers)", lambda n, d, g: n.startswith(("_gqa_sparse", "_merge_topk", "fmhaSm100"))),
        ("lm_head", lambda n, d, g: n.startswith("nvjet")),
    ],
}
comp = defaultdict(float); n = len(pairs); period = 0
for s, nxt in pairs:
    period += (nxt[0][0] - s[0][0]) / 1e3 / n
    last = s[0][0]
    for r in sorted(s, key=lambda r: r[1]):
        e = max(0, r[1] - max(last, r[0])) / 1e3 / n
        last = max(last, r[1])
        g = f"{r[4]}x{r[5]}"
        name = next((k for k, f in RULES[eng] if f(r[2], r[3], g)), "other (sampling, bookkeeping)")
        comp[name] += e
comp["gaps (no kernel running)"] = period - sum(comp.values())
print(f"# {eng}: {n} decode steps, period {period:.1f} us")
order = [k for k, _ in RULES[eng]] + ["other (sampling, bookkeeping)", "gaps (no kernel running)"]
for k in order:
    print(f"{k}\t{comp[k]:.1f}")
