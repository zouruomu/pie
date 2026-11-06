# Gemma-4-26B c512 per run: tok/s, TTFT p50/p99, ITL p50/p99, latency p99, requests with a mid-request stall (> 300 ms after token 10), worst stall.
import json, sys
print(f"{'run':34s} {'tok/s':>6s} {'TTFT p50':>9s} {'p99':>6s} {'ITL p50':>8s} {'p99':>6s} {'lat p99':>8s} {'req w/ mid stall':>17s} {'worst':>6s}")
for f in sys.argv[1:]:
    d = json.load(open(f)); s = d['summary']; rq = [r for r in d['requests'] if r['ok']]
    n = sum(1 for r in rq if any(g > 300000 for g in r['intertoken_us'][10:]))
    w = max(max(r['intertoken_us'][10:]) for r in rq)
    print(f"{f.split('/')[-1][:-5]:34s} {s['output_tok_per_s']:6.0f} {s['ttft_p50_ms']:9.0f} {s['ttft_p99_ms']:6.0f} {s['intertoken_p50_ms']:8.1f} {s['intertoken_p99_ms']:6.0f} {s['latency_p99_ms']:8.0f} {n:10d} ({100*n/len(rq):4.1f}%) {w/1e3:6.0f}")
