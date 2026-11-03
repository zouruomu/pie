# Per-step breakdown of an SGLang nsys node-trace sqlite export.
import sqlite3, sys, statistics as st
from collections import defaultdict
db = sys.argv[1]
c = sqlite3.connect(db)
rows = c.execute("""select k.start, k.end, s.value, k.gridX, k.gridY, k.blockX, k.blockY, d.value, k.graphId, k.streamId
 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName join StringIds d on d.id=k.demangledName order by k.start""").fetchall()
starts = [i for i, r in enumerate(rows) if r[2] == "indexSelectSmallIndex"]
steps = []
for a, b in zip(starts, starts[1:]):
    ks = rows[a:b]
    if any("chunk" in r[2] or r[2] == "BatchPrefillWithPagedKVCacheKernel" and r[3] != 128 for r in ks):
        continue
    steps.append((ks, rows[b][0] - rows[a][0]))
print("decode steps:", len(steps))

def classify(k, prev):
    nm, gx, gy, bx, by, dm, g = k[2], k[3], k[4], k[5], k[6], k[7], k[8]
    if g is None: return "eager", "scheduler/sampler eager kernels"
    if nm == "indexSelectSmallIndex": return "embed", "embedding gather"
    if nm == "kernel" and gx == 31040: return "head", "lm_head gemv (cuBLAS)"
    if nm == "softcap_inplace_logits_kernel": return "head", "softcap (no-op cap)"
    if nm == "kernel" and gx == 2048: return "gdn", "in_proj qkvz gemv (cuBLAS 8192x1024)"
    if nm == "kernel" and gx == 8: return "gdn", "in_proj ba gemv (alt stream, overlapped)"
    if nm.startswith("_fused_qkvzba_causal_conv1d"): return "gdn", "fused unpack+conv1d update"
    if nm.startswith("fused_recurrent_gated_delta"): return "gdn", "gated delta step (+gates, l2norm)"
    if nm == "_layer_norm_fwd_1pass_kernel": return "gdn", "gated rmsnorm"
    if nm == "kernel" and gx == 1280: return "attn", "qkv proj gemv (cuBLAS 5120x1024)"
    if nm.startswith("_fused_qk_rmsnorm_rope_gate"): return "attn", "q/k norm + rope (+gate)"
    if nm == "store_kvcache_kernel": return "attn", "kv store"
    if nm == "BatchPrefillWithPagedKVCacheKernel": return "attn", "flashinfer attention (prefill kernel used for decode)"
    if nm == "PersistentVariableLengthMergeStatesKernel": return "attn", "merge states"
    if nm == "_fused_sigmoid_mul_kernel": return "attn", "sigmoid gate mul"
    if nm == "kernel" and gx == 1792: return "mlp", "gate_up gemv (cuBLAS 7168x1024)"
    if nm == "act_and_mul_kernel": return "mlp", "silu*mul"
    if nm == "kernel" and gx == 256:
        if prev == "_layer_norm_fwd_1pass_kernel": return "gdn", "out_proj gemv (cuBLAS 1024x2048)"
        if prev == "_fused_sigmoid_mul_kernel": return "attn", "o_proj gemv (cuBLAS 1024x2048)"
        return "mlp", "down gemv (cuBLAS 1024x3584)"
    if "FusedAddRMSNorm" in dm or "RMSNormKernel" in dm: return "norm", "fused add+rmsnorm"
    if nm == "reduce_kernel": return "sample", "argmax"
    return "other", nm

per = defaultdict(list); ncall = defaultdict(int); ksum = []; busy = []; periods = []
for ks, period in steps:
    tot = defaultdict(float); prev = None
    for k in ks:
        cat, lab = classify(k, prev)
        if k[8] is not None and k[2] != "kernel" or k[2] == "kernel":
            pass
        tot[(cat, lab)] += (k[1] - k[0]) / 1000
        ncall[(cat, lab)] += 1
        if k[8] is not None and not (k[2] == "kernel" and k[3] == 8): prev = k[2]
    for key, v in tot.items(): per[key].append(v)
    # union of busy intervals
    iv = sorted((k[0], k[1]) for k in ks)
    u = 0; cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce: u += ce - cs; cs, ce = s, e
        else: ce = max(ce, e)
    u += ce - cs
    busy.append(u / 1000); periods.append(period / 1000)
order = ["embed", "gdn", "attn", "mlp", "norm", "head", "sample", "eager", "other"]
cats = defaultdict(float)
for (cat, lab), v in sorted(per.items(), key=lambda kv: (order.index(kv[0][0]), -st.mean(kv[1]))):
    m = sum(v) / len(steps); cats[cat] += m
    n = ncall[(cat, lab)] / len(steps); print(f"{cat:7s} {lab:52s} {n:5.1f}/step {m/n:6.2f} us/call {m:8.1f} us/step")
print("--- by category (sum of kernel durations, incl. overlapped)")
for cat in order:
    if cat in cats: print(f"{cat:7s} {cats[cat]:8.1f}")
print("sum %.1f  busy-union median %.1f  period median %.1f" % (sum(cats.values()), st.median(busy), st.median(periods)))
