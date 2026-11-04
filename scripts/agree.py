# Greedy-token agreement between runs: per request, the share of tokens before the first difference.
# usage: agree.py a.json b.json [...pairs]; prints identical requests and mean/min common-prefix share.
import json, sys, statistics as st
args = sys.argv[1:]
for a, b in zip(args[::2], args[1::2]):
    ra = json.load(open(a))["requests"]; rb = json.load(open(b))["requests"]
    shares, same = [], 0
    for x, y in zip(ra, rb):
        x, y = x.get("output_token_ids"), y.get("output_token_ids")
        if x is None or y is None:
            continue
        p = next((k for k, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
        shares.append(p / max(len(x), 1)); same += x == y
    if shares:
        print(f"{a.split('/')[-1]:34s} vs {b.split('/')[-1]:34s} identical {same}/{len(shares)}  "
              f"prefix share mean {st.mean(shares):.3f} min {min(shares):.3f}")
