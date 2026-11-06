# Summarize pie pbatch JSONs: prefill tok/s = prompt tokens / wall; TTFT p50 = latency p50 (1 output token).
import json, sys, glob, os
for f in sorted(sys.argv[1:], key=lambda f: (f.split('-')[1], int(f.split('-pf')[1].split('-')[0].split('.')[0]), f)):
    try: d = json.load(open(f))['summary']
    except Exception as e: print(os.path.basename(f), 'ERR', e); continue
    ck = open(f[:-5] + '.clocks').read().split('\n')[1] if os.path.exists(f[:-5] + '.clocks') else ''
    print(f"{os.path.basename(f):32s} {d['prompt_tokens']:7d} {d['wall_s']:7.3f}s {d['prompt_tokens']/d['wall_s']:9.0f} tok/s  p50 {d['latency_p50_ms']:8.1f} ms  [{ck}]")
