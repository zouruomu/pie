# Per-request token gaps from bench JSONs: median TTFT (from the client's seat), median gap between
# tokens 1 and 2, median of the later gaps, and the mean time a request is live without a token
# (latency - (n-1) x median later gap), per file.
import json
import statistics
import sys

for f in sys.argv[1:]:
    rs = json.load(open(f))["requests"]
    first, later, idle = [], [], []
    for r in rs:
        arr = r.get("token_arrival_s")
        if not arr or len(arr) < 3:
            continue
        gaps = [b - a for a, b in zip(arr, arr[1:])]
        first.append(gaps[0])
        med = statistics.median(gaps[1:])
        later.append(med)
        idle.append(r["latency_s"] - (len(arr) - 1) * med)
    if not first:
        print(f, "no arrivals")
        continue
    ttft = statistics.median(r["ttft_s"] for r in rs if r.get("ttft_s") is not None)
    print(f"{f.split('/')[-1]:36s} n={len(first):5d} ttft50={ttft:7.2f}s gap1-2 p50={statistics.median(first) * 1e3:7.1f}ms "
          f"p90={sorted(first)[int(0.9 * len(first))] * 1e3:7.1f}ms later p50={statistics.median(later) * 1e3:5.1f}ms "
          f"idle/request mean={statistics.mean(idle):5.2f}s")
