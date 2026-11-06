# Compare full greedy token-id lists of two bench JSONs, request by request.
import json, sys
def ids(p):
    d = json.load(open(p)); return [r.get("all_token_ids") or r.get("output_token_ids") or r.get("token_ids") for r in d["requests"]]
a, b = ids(sys.argv[1]), ids(sys.argv[2])
for i, (x, y) in enumerate(zip(a, b)):
    if x is None or y is None: print(i, "no ids"); continue
    d = next((j for j, (u, v) in enumerate(zip(x, y)) if u != v), None)
    print(f"req {i}: len {len(x)}/{len(y)} " + ("identical" if d is None and len(x) == len(y) else f"first diff at {d}"))
