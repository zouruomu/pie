# Note 1

Good early verdict. Before you call batching green, close the gap between your setup and the site's:

1. Your best vLLM point is 5197; the site has 7198 for vLLM, 38% more. Your pie numbers also differ from the site's (7437 vs 4560). Reproduce the site's own harness invocation exactly, for every engine, from `.github/workflows/evals.yml` and `scripts/bench/*` (and anton-mel/stats if it drives them): offline vs streaming mode, number of requests a point, `--report-timing`, warmup. Note `scripts/bench/vllm_bench.py` sets `max_num_seqs = max(1, num_requests)` in one path and `= concurrency` in the other (see lines ~126-130 and ~372-376, and open PR #1437 which caps it). If the site's vLLM runs all requests at once, its "c" label isn't the lanes it ran.
2. Report both views in status.md: site-identical flags (what the site scores), and the fair, matched-lanes comparison. A pie gap under the site's flags that comes from the harness, not the engine, is a side finding for the report, not an engine fix. Say which one it is.
3. Keep raw prefill and pf32–pf256 against SGLang next as planned. If SGLang still OOMs at load, use vLLM there and say so; don't spend more than ~30 min on SGLang's load.
