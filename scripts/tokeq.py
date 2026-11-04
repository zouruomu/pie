# Compare full greedy token-id lists, request by request (results are in request-index order),
# between pairs of bench JSONs: tokeq.py a.json b.json [a2.json b2.json ...]
import json, sys
args = sys.argv[1:]
allsame = True
for a, b in zip(args[::2], args[1::2]):
    ra = json.load(open(a))["requests"]; rb = json.load(open(b))["requests"]
    ids_a = [r.get("output_token_ids") for r in ra]; ids_b = [r.get("output_token_ids") for r in rb]
    if any(x is None for x in ids_a + ids_b):
        print(f"{a} vs {b}: missing token ids"); allsame = False; continue
    diff = [i for i, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y]
    toks = sum(len(x) for x in ids_a)
    first = ""
    if diff:
        i = diff[0]; x, y = ids_a[i], ids_b[i]
        p = next((k for k, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
        first = f" first diff: request {i} at token {p} of {len(x)}"
        allsame = False
    print(f"{a.split('/')[-1]} vs {b.split('/')[-1]}: {len(ids_a)} requests, {toks} tokens, "
          f"{len(diff)} requests differ{first}")
print("ALL IDENTICAL" if allsame else "DIFFERENCES FOUND")
