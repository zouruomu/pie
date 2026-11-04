# Summarise bench JSONs: label, wall, prompt/output tokens, prefill tok/s (sum prompt / wall),
# output tok/s, TTFT p50/p99, latency p50, ITL p50, SM clock at end, engine sha.
import json, sys, glob, os
rows = []
for f in sorted(sum((glob.glob(a) for a in sys.argv[1:]), [])):
    try:
        d = json.load(open(f))
    except Exception as e:
        print(os.path.basename(f), "BAD", e); continue
    s = d["summary"]; c = s.get("config") or {}
    w = s["wall_s"]; pt = s.get("prompt_tokens") or 0
    clk = c.get("gpu clocks at end") or [{}]
    print(f"{os.path.basename(f)[:-5]:34s} ok={s['completed']}/{s['requests']} wall={w:8.2f} "
          f"prefill={pt / w if w else 0:8.1f} out={s['output_tok_per_s']:8.1f} "
          f"ttft50={s.get('ttft_p50_ms') or 0:9.1f} ttft99={s.get('ttft_p99_ms') or 0:9.1f} "
          f"lat50={s.get('latency_p50_ms') or 0:9.1f} itl50={s.get('intertoken_p50_ms') or 0:6.1f} "
          f"sm={clk[0].get('sm_mhz')} sha={(c.get('embedded engine sha256') or '')[:8]}")
