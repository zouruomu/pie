# Traced runs: pool low point, pauses/restores, alloc-queue waits; and the stall each paused process (pid = the JSON's process_id) saw.
import json, re, sys
for base in sys.argv[1:]:
    log = re.sub(r'\x1b\[[0-9;]*m', '', open(base+'.log').read()).splitlines()
    rep = [l for l in log if l.startswith('[report]')]
    free = [int(re.search(r'pages free=(\d+)', l).group(1)) for l in rep]
    infl = [int(re.search(r'inflight=(\d+)', l).group(1)) for l in rep]
    alloc_waits = sum(l.count(' alloc kv=') for l in rep); restores = sum(l.count(' restore kv=') for l in rep)
    paused = [l for l in log if 'pie::kv: paused' in l]
    ppids = set(re.search(r'pid=(\S+)', l).group(1) for l in log if 'to HostPinned' in l)
    d = json.load(open(base+'.json')); rq = {r['process_id']: r for r in d['requests']}
    pg = sorted(max(rq[p]['intertoken_us'])/1e3 for p in ppids if p in rq)
    allg = sorted(max(r['intertoken_us'][10:]) / 1e3 for r in d['requests'])
    print(f"{base:30s} min free {min(free)} of 5887, max inflight {max(infl)}; paused {len(paused)} ({len(ppids)} processes, KV to host and back); "
          f"queued restores {restores}, queued page allocs {alloc_waits} (report lines); paused processes' worst gap ms {pg[:1]}..{pg[-1:]}; "
          f"sealed eager fires {sum('sealed map holds no body' in l for l in log)}")
