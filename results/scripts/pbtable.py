# Per-point prefill tok/s (median over rounds, min-max) for pie, TRT-LLM and vLLM from results/<dir>.
# usage: pbtable.py <dir> [pie-prefix]
import glob, json, os, re, statistics, sys
from collections import defaultdict

d = sys.argv[1]
pie_prefix = sys.argv[2] if len(sys.argv) > 2 else "pie"
v = defaultdict(list)  # (engine, model, n) -> [(tok/s, prompt tokens a request, ttft p50 ms)]
for f in glob.glob(f"{d}/*.json"):
    b = os.path.basename(f)
    m = re.match(r"(\w+?)-(20b|120b)-pf(\d+)-r\d+v?\.json$", b)
    if m and m.group(1) in (pie_prefix, "vllm"):
        s = json.load(open(f))["summary"]
        eng = "pie" if m.group(1) == pie_prefix else "vllm"
        v[(eng, m.group(2), int(m.group(3)))].append(
            (s["prompt_tokens"] / s["wall_s"], s["prompt_tokens"] / s["completed"], s["latency_p50_ms"]))
    m = re.match(r"trt-(20b|120b)-(sweep|big)-r\d+\.json$", b)
    if m:
        for r in json.load(open(f))["rows"]:
            v[("trt", m.group(1), r["requests"])].append(
                (r["prefill_tok_s"], r["prompt_tokens"] / r["requests"], r["ttft_p50_ms"]))


def cell(xs):
    if not xs:
        return "", None
    t = [x[0] for x in xs]
    med = statistics.median(t)
    spread = f" ({min(t)/1e3:.1f}-{max(t)/1e3:.1f}, n={len(t)})" if len(t) > 1 else " (n=1)"
    return f"{med/1e3:.1f}k{spread}", med


for model in ("20b", "120b"):
    print(f"\n### gpt-oss-{model}\n")
    print("| pf | pie | TRT-LLM | vLLM | pie vs TRT-LLM | pie vs vLLM | pie TTFT p50 ms | TRT-LLM TTFT p50 ms |")
    print("|---|---|---|---|---|---|---|---|")
    for n in (4, 8, 16, 32, 64, 128, 256, 512, 1024):
        (ps, p), (ts, t), (vs, vv) = (cell(v[(e, model, n)]) for e in ("pie", "trt", "vllm"))
        g = lambda a, b: f"{(a / b - 1) * 100:+.1f}%" if a and b else ""
        pt = statistics.median(x[2] for x in v[("pie", model, n)]) if v[("pie", model, n)] else None
        tt = statistics.median(x[2] for x in v[("trt", model, n)]) if v[("trt", model, n)] else None
        print(f"| {n} | {ps} | {ts} | {vs} | {g(p, t)} | {g(p, vv)} | {pt and round(pt)} | {tt and round(tt)} |")
    toks = {e: sorted({round(x[1]) for k, xs in v.items() if k[0] == e and k[1] == model for x in xs}) for e in ("pie", "trt", "vllm")}
    print(f"\nprompt tokens a request: {toks}")
