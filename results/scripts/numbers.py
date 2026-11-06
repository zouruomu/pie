# Every headline number from the JSON summaries' exact `intertoken_p50_ms`: per group the rounds, their
# median, decode tok/s = 1000 / median, and the ratios the write-up quotes.
import json, statistics as st
from pathlib import Path
R = Path("/root/r2g/evidence/results/raw")
def itl(name):
    f = R / f"{name}.json"
    return json.load(open(f))["summary"]["intertoken_p50_ms"] if f.exists() else None
groups = {
    "A pie base (baseline)": ["pie-mx8nv-probe", "pie-m31-r2", "pie-m31-r3"],
    "B pie base (baseline)": ["pie-m3-r1", "pie-m3-r2", "pie-m3-r3"],
    "B vLLM (baseline)": ["vllm-m3-r1s", "vllm-m3-r2s", "vllm-m3-r3s"],
    "B SGLang (baseline)": ["sglang-m3-r1", "sglang-m3-r2", "sglang-m3-r3"],
    "A base (final A/B)": [f"fin-a-base-r{r}" for r in (1, 2, 3)],
    "A final": [f"fin-a-final-r{r}" for r in (1, 2, 3)],
    "B base (final A/B)": [f"fin-b-base-r{r}" for r in (1, 2, 3)],
    "B final": [f"fin-b-final-r{r}" for r in (1, 2, 3)],
    "B vLLM (A/B rounds)": [f"ab-b-vllm-r{r}" for r in (1, 2, 3)],
    "Q3.8 base": ["x-q38-base-r1", "x-q38-base-r2"],
    "Q3.8 final": ["x-q38-final-r1", "x-q38-final-r2"],
    "Q0.6 base": ["x-q06-base-r1", "x-q06-base-r2"],
    "Q0.6 final": ["x-q06-final-r1", "x-q06-final-r2"],
}
med = {}
for g, names in groups.items():
    vals = [itl(n) for n in names]
    have = [v for v in vals if v is not None]
    med[g] = st.median(have) if have else None
    shown = " / ".join(f"{v:.3f}" if v is not None else "-" for v in vals)
    tok = f"{1000 / med[g]:.1f} tok/s" if med[g] else ""
    print(f"{g:24s} {shown:30s} median {med[g] if med[g] is None else round(med[g], 3)}  {tok}")
def pct(a, b):
    return f"{(med[a] / med[b] - 1) * 100:+.1f}%" if med[a] and med[b] else "pending"
def gap(p, v):  # pie behind the competitor, in decode tok/s
    return f"{(1 - med[v] / med[p]) * 100:.1f}%" if med[p] and med[v] else "pending"
print("A speedup (tok/s):", pct("A base (final A/B)", "A final"))
print("B speedup (tok/s):", pct("B base (final A/B)", "B final"))
print("B gap vs vLLM, baseline rounds:", gap("B pie base (baseline)", "B vLLM (baseline)"), " vs SGLang:", gap("B pie base (baseline)", "B SGLang (baseline)"))
print("B gap vs vLLM (A/B rounds): base", gap("B base (final A/B)", "B vLLM (A/B rounds)"), "-> final", gap("B final", "B vLLM (A/B rounds)"))
print("Q3.8 speedup:", pct("Q3.8 base", "Q3.8 final"), " Q0.6:", pct("Q0.6 base", "Q0.6 final"))
