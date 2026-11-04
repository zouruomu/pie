# The site's L4 gpt-oss-20b rows (data/db.js, snapshot l4-full) and what they imply:
# prompt tokens a request (prefill/out) and requests (out x wall, wall ~ TTFT p99 or latency p99).
# usage: python3 -I site_rows.py <db.json converted from data/db.js>
import json, sys
db = json.load(open(sys.argv[1]))
rows = [x for x in db["snapshots"]["l4-full"] if x["hardware"] == "l4-x1" and x["model"] == "gpt-oss-20b"
        and x["status"] == "pass"]
for x in rows:
    print(x["engine"], x["version"] or x["commit"], x["workload"], {k: x[k] for k in (
        "prefill_tok_s", "output_tok_s", "decode_tok_s", "ttft_ms_p50", "ttft_ms_p99", "itl_ms_p50",
        "latency_ms_p50", "latency_ms_p99", "measured")})
    if x["workload"] == "prefill_batch":
        print("   prompt tokens a request %.1f, requests ~ %.1f" % (
            x["prefill_tok_s"] / x["output_tok_s"], x["output_tok_s"] * x["ttft_ms_p99"] / 1e3))
    if x["workload"] == "throughput":
        print("   requests if wall ~ latency p99: %.0f; seats if each holds one for latency p50: %.0f" % (
            x["output_tok_s"] * x["latency_ms_p99"] / 1e3 / 128, x["output_tok_s"] * x["latency_ms_p50"] / 1e3 / 128))
