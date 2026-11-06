# Prefill-batch table: Σ prompt tokens / wall (the site's prefill_tok_s) and TTFT p50, per file.
import glob
import json
import sys

for f in sorted(sum((glob.glob(a) for a in sys.argv[1:]), [])):
    s = json.load(open(f))["summary"]
    c = s.get("config") or {}
    print(f"{f.split('/')[-1][:-5]:30s} prefill={s['prompt_tokens'] / s['wall_s']:8.1f} ttft50={s.get('ttft_p50_ms') or 0:8.1f}ms "
          f"sha={(c.get('embedded engine sha256') or '')[:8]}")
