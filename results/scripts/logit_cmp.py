# Compare teacher-forced logits (logit_probe.py output) of base against another build or setting.
# usage: logit_cmp.py <probe dir base> <probe dir other> <base bench json> <other bench json> <label>
# Per request: max |logit diff| and KL(base || other) over the decode positions 1..32 (position 0, after the
# prompt, reported apart: a prefill change shows there); base's argmax checked against the
# teacher (base's own greedy run); at the greedy runs' first diverging position, base's top-2 margin.
import json, sys
from pathlib import Path
import numpy as np
db, do, jb, jo, label = sys.argv[1:6]
tb = [r["output_token_ids"] for r in json.load(open(jb))["requests"]]
to = [r["output_token_ids"] for r in json.load(open(jo))["requests"]]
rows = []
for j in range(4):
    A = np.fromfile(Path(db, f"r{j}.f32"), dtype=np.float32).astype(np.float64).reshape(33, -1)
    B = np.fromfile(Path(do, f"r{j}.f32"), dtype=np.float32).astype(np.float64).reshape(33, -1)
    ks = range(len(A))
    mx, kl, ok, oth, margins = 0.0, [], 0, [], []
    for k in ks:
        a, b = A[k], B[k]
        la = a - a.max(); la -= np.log(np.exp(la).sum())
        lb = b - b.max(); lb -= np.log(np.exp(lb).sum())
        k_l = float((np.exp(la) * (la - lb)).sum())
        if k == 0:
            p0 = (float(np.abs(a - b).max()), k_l)
        else:
            mx = max(mx, float(np.abs(a - b).max()))
            kl.append(k_l)
        ok += int(a.argmax() == tb[j][k])
        oth.append(int(b.argmax()) == tb[j][k])
        top = np.sort(a)[-2:]
        margins.append(float(top[1] - top[0]))
    d = next((i for i, (x, y) in enumerate(zip(tb[j], to[j])) if x != y), None)
    f = next((k for k, same in enumerate(oth) if not same), None)
    rows.append(dict(req=j, positions=len(ks), argmax_matches_teacher=ok, prefill_pos0_max_abs_diff=p0[0],
                     prefill_pos0_kl=p0[1], max_abs_logit_diff=mx,
                     kl_mean=float(np.mean(kl)), kl_max=float(np.max(kl)),
                     other_argmax_matches_teacher=sum(oth),
                     first_teacher_forced_disagreement=f,
                     base_top2_margin_at_it=(margins[f] if f is not None else None),
                     first_greedy_divergence=d,
                     base_top2_margin_there=(margins[d] if d is not None and d < len(margins) else None),
                     base_top2_margin_median=float(np.median(margins))))
print(f"# {label}")
for r in rows:
    print(json.dumps(r))
