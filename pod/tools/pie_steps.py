# Per-step breakdown of a Pie nsys node-trace sqlite export.
# Steps are delimited by the step's first graph kernel (the embed row_skeleton
# that follows a `settle`); a step runs to the next step's start.
import sqlite3, sys, statistics as st
from collections import defaultdict, OrderedDict
db = sys.argv[1]
c = sqlite3.connect(db)
rows = c.execute("""select k.start, k.end, s.value, k.gridX, k.gridY, k.blockX, k.blockY, d.value, k.graphId
 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName join StringIds d on d.id=k.demangledName order by k.start""").fetchall()
starts = [i for i in range(1, len(rows)) if rows[i-1][2] == "settle"]
steps = []
for a, b in zip(starts, starts[1:]):
    ks = rows[a:b]
    if b - a != 285 or any(r[2].startswith("gdn_chunk") for r in ks):
        continue
    steps.append(ks)
print("decode steps:", len(steps))

def classify(k, layer_kind):
    nm, gx, gy, bx, dm = k[2], k[3], k[4], k[5], k[7]
    if nm == "gemv_skeleton" and gx == 62080: return "head", "lm_head gemv (248320x1024)"
    if nm == "gather_rows": return "head", "gather_rows (readout row)"
    if nm in ("set_conditional", "pull_validate", "commit_bump", "scatter_publish", "settle"): return "epilogue", nm
    if nm.startswith("ptir_fused"): return "epilogue", "argmax sample (ptir_fused)"
    if nm == "gemv_skeleton" and gx == 2048: return "gdn", "in_proj qkv+z gemv (8192x1024)"
    if nm == "split_rows_vec8": return "gdn", "split qkv/z"
    if nm == "gemv_splitk_skeleton" and gx == 32: return "gdn", "in_proj b,a gemv (32x1024)"
    if nm.startswith("conv1d_update"): return "gdn", "causal conv1d update"
    if nm == "gdn_prep_ba_gates": return "gdn", "gdn gates"
    if nm == "gdn_step_fused": return "gdn", "gated delta step"
    if nm == "rmsnorm_gated_in_warp": return "gdn", "gated rmsnorm"
    if nm == "kernel" and "gemvx" in dm: return "attn", "q/gate+k+v proj (cuBLAS gemvx 5120x1024)"
    if nm == "row_skeleton" and bx == 1024: return "attn", "q/k norm + rope + kv append (row_skeleton)"
    if nm == "BatchDecodeWithPagedKVCacheKernel": return "attn", "fa2 decode attention"
    if nm == "PersistentVariableLengthMergeStatesKernel": return "attn", "merge states"
    if nm == "gate_sigmoid_mul_vec8": return "attn", "sigmoid gate mul"
    if nm == "gemv_splitk_skeleton" and gx == 1024 and ", (int)1, (int)1, (int)0," in dm:
        return layer_kind, ("out_proj gemv (1024x2048)" if layer_kind == "gdn" else "o_proj gemv (1024x2048)")
    if nm == "gemv_splitk_skeleton" and gx == 3584: return "mlp", "gate+up+silu gemv (2x3584x1024)"
    if nm == "gemv_splitk_skeleton" and gx == 1024: return "mlp", "down gemv (+residual) (1024x3584)"
    if nm == "rmsnorm_runs": return "norm", "rmsnorm_runs (resid+norm)"
    if nm == "row_skeleton": return "norm", "row_skeleton 128t (embed/resid+norm)"
    return "other", nm

per = defaultdict(list); ncall = defaultdict(int)
cnt = defaultdict(int)
spans, ksums, gsum_in, gsum_out = [], [], [], []
for s in steps:
    tot = defaultdict(float)
    layer = "gdn"
    for k in s:
        if k[2] == "gemv_skeleton" and k[3] == 2048: layer = "gdn"
        if k[2] == "kernel": layer = "attn"
        cat, lab = classify(k, layer)
        tot[(cat, lab)] += (k[1] - k[0]) / 1000
        ncall[(cat, lab)] += 1
    for key, v in tot.items():
        per[key].append(v)
    ksums.append(sum((k[1]-k[0]) for k in s) / 1000)
for s, nxt in zip(steps, steps[1:]):
    pass
# step period: start of step i to start of step i+1 where consecutive
idx = {id(s[0]): i for i, s in enumerate(steps)}
periods = []
for a, b in zip(starts, starts[1:]):
    pass
print("kernel-time per step: median %.1f us" % st.median(ksums))
order = ["embed", "gdn", "attn", "mlp", "norm", "head", "epilogue", "other"]
cats = defaultdict(float)
for (cat, lab), v in sorted(per.items(), key=lambda kv: (order.index(kv[0][0]), -st.mean(kv[1]))):
    m = st.mean(v); cats[cat] += m
    n = sum(1 for k in steps[0] if True)  # placeholder
    n = ncall[(cat, lab)] / len(steps); print(f"{cat:9s} {lab:48s} {n:5.1f}/step {m/n:6.2f} us/call {m:8.1f} us/step")
print("--- by category")
for cat in order:
    if cat in cats: print(f"{cat:9s} {cats[cat]:8.1f}")
print("total", sum(cats.values()))
