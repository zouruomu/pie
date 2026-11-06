# Mid-request gaps (> 300 ms after token 10): a global stall (no one emits during it) or this lane set aside while others decode?
import json, sys, bisect
for f in sys.argv[1:]:
    d = json.load(open(f)); rq = [r for r in d['requests'] if r['ok']]
    allt = sorted(t for r in rq for t in r['token_arrival_s'][1:])
    firsts = sorted(r['token_arrival_s'][0] for r in rq)
    mids = []; lanes = set()
    for k, r in enumerate(rq):
        a = r['token_arrival_s']
        for i in range(11, len(a)):
            if a[i]-a[i-1] > 0.3:
                lo, hi = a[i-1], a[i]
                others = bisect.bisect_left(allt, hi) - bisect.bisect_right(allt, lo)
                nf = bisect.bisect_left(firsts, hi) - bisect.bisect_right(firsts, lo)
                mids.append((lo, hi-lo, others/(hi-lo), nf, i)); lanes.add(k)
    if not mids: print(f.split('/')[-1], 'no mid gaps'); continue
    mids.sort()
    durs = sorted(m[1] for m in mids); rates = sorted(m[2] for m in mids)
    print(f"{f.split('/')[-1][:-5]:30s} mid gaps {len(mids)} in {len(lanes)} requests; dur med {durs[len(durs)//2]*1e3:.0f} max {durs[-1]*1e3:.0f} ms; "
          f"other tokens/s during gap: min {rates[0]:.0f} med {rates[len(rates)//2]:.0f}; with 0 firsts inside: {sum(1 for m in mids if m[3]==0)}")
    # cluster by start (0.25 s buckets)
    b = {}
    for lo, du, ra, nf, i in mids: b.setdefault(round(lo*4)/4, []).append((du, nf, i))
    print('   start: n gaps, med dur ms, firsts inside, token pos range ->', '; '.join(f"{k:.2f}s: {len(v)}, {sorted(x[0] for x in v)[len(v)//2]*1e3:.0f}, {sorted(x[1] for x in v)[len(v)//2]}, {min(x[2] for x in v)}-{max(x[2] for x in v)}" for k, v in sorted(b.items())[:12]))
