# Review 1 item 1: where do Gemma-4-26B inter-token gaps over 300 ms fall?
# Per run: ITL p99, gaps > 300 ms (count, token position, run time), and whether each one spans other requests' first tokens (a prefill wave).
import json, sys, bisect
TH = 0.300
for f in sys.argv[1:]:
    d = json.load(open(f)); rq = [r for r in d['requests'] if r['ok']]
    itl = sorted(g for r in rq for g in r['intertoken_us'])
    p = lambda q: itl[min(len(itl)-1, int(q*len(itl)))]/1e3
    firsts = sorted(r['token_arrival_s'][0] for r in rq)
    big = []
    for r in rq:
        a = r['token_arrival_s']
        for i in range(1, len(a)):
            if a[i]-a[i-1] > TH:
                lo, hi = a[i-1], a[i]
                n_first = bisect.bisect_left(firsts, hi) - bisect.bisect_right(firsts, lo)
                big.append((i, lo, hi-lo, n_first))
    pos = {}
    for i, *_ in big: k = '1->2' if i == 1 else ('2-10' if i <= 10 else '>10'); pos[k] = pos.get(k, 0)+1
    starts = sorted(round(lo, 1) for _, lo, _, _ in big)
    waves = sorted(set(starts))
    spans = [b[3] for b in big]
    s = d['summary']
    print(f"{f.split('/')[-1][:-5]:34s} tok/s {s['output_tok_per_s']:6.0f} wall {s['wall_s']:5.1f}s ITL p50 {p(.5):5.0f} p99 {p(.99):6.0f} max {itl[-1]/1e3:6.0f} ms | "
          f">300ms: {len(big):4d} of {len(itl)} ({100*len(big)/len(itl):.2f}%) pos {pos} "
          f"| span other firsts: min {min(spans) if spans else '-'} med {sorted(spans)[len(spans)//2] if spans else '-'} "
          f"| distinct start times (0.1s) {len(waves)}: {waves[:8]}")
