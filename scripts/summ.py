import json,sys,glob,os
for f in sorted(sys.argv[1:]):
    try: d=json.load(open(f))
    except Exception as e: print(f,'ERR',e); continue
    s=d['summary'];c=s.get('config',{})
    print(f"{os.path.basename(f):28s} {s['output_tok_per_s']:8.0f} tok/s  in={s['prompt_tokens']//max(1,s['completed'])} n={s['completed']} wall={s['wall_s']:.1f} ttft50={s.get('ttft_p50_ms')} itl50={s.get('intertoken_p50_ms')} maxfr={c.get('max forward requests')} act={c.get('wave avg active pipelines')} sha={str(c.get('embedded engine sha256'))[:8]}")
