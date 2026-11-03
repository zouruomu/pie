"""List the cells where pie trails the fastest other engine on benchmarks.pie-project.org.

Mirrors the site's platform grid (assets/js/ui/platform.js, core/store.js): the newest
release's snapshot, measured rows only (status pass or noisy), and for each machine,
model and mode, gain = pie / best other engine - 1. A cell is red when gain <= -0.045.

usage: python3 red_cells.py [db.js path or URL] [--json out.json]
"""
import json
import sys
import urllib.request

URL = "https://benchmarks.pie-project.org/data/db.js"
RED = -0.045
MODES = {  # mode: (workload, metric key)
    "decode": ("raw", "decode_tok_s"),
    "prefill": ("raw", "prefill_tok_s"),
    "batching": ("throughput", "output_tok_s"),
    "pbatch": ("prefill_batch", "prefill_tok_s"),
}


def load(src):
    if src.startswith("http"):
        with urllib.request.urlopen(src) as r:
            text = r.read().decode()
    else:
        text = open(src).read()
    return json.loads(text[text.index("{"):text.rindex("}") + 1])


def red_cells(db):
    newest = next(t["id"] for t in db["tags"] if db["snapshots"].get(t["id"]))
    rows = [r for r in db["snapshots"][newest] if not r.get("status") or r["status"] in ("pass", "noisy")]
    hardware = {h["id"]: h for h in db["hardware"]}
    cells = {}
    for r in rows:
        for mode, (workload, key) in MODES.items():
            if r["workload"] == workload and r.get(key):
                cells.setdefault((r["hardware"], r["model"], mode), []).append(r)
    out = []
    for (mac, model, mode), rs in cells.items():
        key = MODES[mode][1]
        pie = next((r for r in rs if r["engine"] == "pie"), None)
        others = [r for r in rs if r["engine"] != "pie"]
        if not pie or not others:
            continue
        best = max(others, key=lambda r: r[key])
        gain = pie[key] / best[key] - 1
        if gain > RED:
            continue
        hw = hardware.get(mac, {})
        out.append({
            "hardware": mac, "gpu": hw.get("full_name"), "gpu_count": hw.get("count"), "model": model,
            "mode": mode, "gain": round(gain, 4), "pie": pie[key], "best_engine": best["engine"],
            "best": best[key], "scheme": pie.get("scheme"), "miniature": bool(pie.get("miniature")),
            "pie_commit": pie.get("commit"), "pie_measured": pie.get("measured"),
            "pie_stale": pie.get("stale"), "best_version": best.get("version"),
            "best_measured": best.get("measured"),
        })
    return newest, sorted(out, key=lambda c: c["gain"])


def main():
    args = sys.argv[1:]
    out_json = args[args.index("--json") + 1] if "--json" in args else None
    src = next((a for a in args if a != "--json" and a != out_json), URL)
    db = load(src)
    newest, cells = red_cells(db)
    print(f"db updated {db.get('updated_at')}, newest release {newest}, {len(cells)} red cells")
    for c in cells:
        print(f"{c['gain']:+7.1%}  {c['hardware']:<14} {c['model']:<34} {c['mode']:<9} "
              f"pie {c['pie']:<9} {c['best_engine']} {c['best']:<9} pie@{c['pie_commit']} {c['pie_measured']}"
              f"{'  mini' if c['miniature'] else ''}{'  stale' if c['pie_stale'] else ''}")
    if out_json:
        json.dump({"updated_at": db.get("updated_at"), "release": newest, "cells": cells}, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
