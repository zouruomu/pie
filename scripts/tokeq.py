# Compare full greedy token-id lists, request by request (results are in request-index order),
# between pairs of bench JSONs: tokeq.py a.json b.json [a2.json b2.json ...]
# Prints, per pair: requests compared, tokens, requests that differ, the first differing request and
# position, and the share of all tokens before each request's first divergence.
import json
import sys

args = sys.argv[1:]
for a, b in zip(args[::2], args[1::2]):
    ra = json.load(open(a))["requests"]
    rb = json.load(open(b))["requests"]
    ids_a = [r.get("output_token_ids") for r in ra]
    ids_b = [r.get("output_token_ids") for r in rb]
    if any(x is None for x in ids_a + ids_b):
        print(f"{a} vs {b}: missing token ids")
        continue
    toks = sum(len(x) for x in ids_a)
    diff, agree, first = 0, 0, ""
    for i, (x, y) in enumerate(zip(ids_a, ids_b)):
        p = next((k for k, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
        agree += p
        if x != y:
            diff += 1
            if not first:
                first = f", first: request {i} at token {p}"
    print(f"{a.split('/')[-1]} vs {b.split('/')[-1]}: {len(ids_a)} requests, {toks} tokens, "
          f"{diff} differ{first}, {agree / toks:.1%} of tokens before divergence")
