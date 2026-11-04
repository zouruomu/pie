# The site's metric: each engine's best point of one sweep (batching c4..c1024, pbatch pf4..pf1024),
# per round, then mean and spread over rounds. usage: best.py name=prefix[,prefix...] ...
import glob, json, re, statistics as st, sys
for arg in sys.argv[1:]:
    name, prefixes = arg.split("=", 1)
    rounds = {}
    for p in prefixes.split(","):
        for f in glob.glob(f"{p}-*-r*.json"):
            m = re.match(re.escape(p) + r"-(c\d+|pf\d+)-r(\d+)\.json$", f)
            if not m:
                continue
            d = json.load(open(f)); s = d["summary"]
            if s["completed"] != s["requests"]:
                continue
            shape, r = m.group(1), (p, m.group(2))
            v = s["output_tok_per_s"] if shape.startswith("c") else s["prompt_tokens"] / s["wall_s"]
            kind = "batching" if shape.startswith("c") else "pbatch"
            rounds.setdefault((r, kind), []).append((v, shape))
    for kind in ("batching", "pbatch"):
        best = [max(v) for (r, k), v in sorted(rounds.items()) if k == kind]
        if best:
            vals = [b[0] for b in best]
            print(f"{name:6s} {kind:9s} best per round: " + ", ".join(f"{v:.0f} ({s})" for v, s in best)
                  + f"  mean {st.mean(vals):.0f}  spread {min(vals):.0f}-{max(vals):.0f}")
