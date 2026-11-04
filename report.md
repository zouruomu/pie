# L4 × gpt-oss-20b (mxfp4): batching and pbatch vs vLLM 0.32.0

Run `20261103-2218-l4-x1-gpt-oss-20b-batching`, one NVIDIA L4 (24 GB, sm_89, 72 W cap) on RunPod.
Fix branch `r2g/20261103-2218-l4-x1-gpt-oss-20b-batching`, evidence branch `r2g-evidence/…`.

DRAFT — numbers marked TBD are filled from the verification sweep.

## 1. Summary

TBD

## 2. How the site computes each metric

The GPU harness (pie-project/benchmarks) is not readable with this token (`gh repo clone` → "Could not
resolve to a Repository"). What it reports was reconstructed from three sources:

* `scripts/bench/common.py:211-255` (`summarize`): `output_tok_per_s = Σ output tokens / wall`,
  TTFT/latency percentiles over requests.
* anton-mel/stats `src/pie_evals/node/engines/base.py:100-150` (`perf_from_common_json`, the Mac
  harness that drives the same `common.py` client): `prefill_tok_s` = median over requests of
  `prompt_tokens / ttft`, `decode_tok_s` = median of `(n-1)/(latency-ttft)`.
* The site's own rows (`data/db.js`, snapshot `l4-full`, db updated 2026-11-03T22:01Z), checked for
  internal consistency:

| row | prefill_tok_s / output_tok_s | output_tok_s × wall (wall ≈ TTFT p99) | reading |
|---|---|---|---|
| pie pbatch | 4666.45 / 7.66 = **609.2** tokens a request | 7.66 × 1.057 s = **8.1** requests | **pf8** |
| vLLM pbatch | 6424.65 / 10.58 = **607.2** | 10.58 × 6.043 s = **63.9** | **pf64** |
| SGLang pbatch | 558.40 / 0.92 = **607.0** | 0.92 × 34.29 s = **31.5** | **pf32** |

So for `prefill_batch` the site's `prefill_tok_s` is `Σ prompt tokens / wall` (the per-request
`prompt/ttft` median would be 609/1.045 s = 583 tok/s, not 4666), the prompt is ~609 chat-rendered
tokens (512 + gpt-oss's harmony preamble), and **each engine's cell is its own best pf-N**: pie's
pf8, vLLM's pf64, SGLang's pf32.

For `throughput` (batching): pie 1576.34 tok/s with latency p50 40.1 s → each request holds a seat
~41.6 s → 1576×41.6/128 ≈ 512 in flight → **c512** (1024 requests, wall ≈ 83 s). vLLM 2192.91 tok/s
with latency p99 119.5 s; vLLM's client submits every request at t=0 (`vllm_bench.py:538`), so its
wall ≈ p99 latency → 2192.91×119.5/128 = **2047 requests → c1024** (2×1024). Again each engine's best
point. Note the two engines' latencies are not comparable: pie's client holds `--concurrency` open
itself (`pie_bench.py:1100-1110`, the request's clock starts when it gets a seat), vLLM's queues in
the engine (`max_num_seqs = concurrency`, `vllm_bench.py:373-376`) and its TTFT includes that queue.
The brief's "pie has better latency and ITL" is this, not an engine property.

`raw` for pie: 4680.97 tok/s × 128.93 ms = 603.5 tokens, i.e. `prompt_tokens/ttft` per request.

Reproduction here: prompts built with `scripts/make_prompts.py` to exactly 225 (batching) and 609
(pbatch, raw) chat-rendered tokens with the HF tokenizer; pie runs `--pretokenized-prompts`, so both
engines prefill the identical token ids (pie's own harmony rendering is 545 tokens for the same text,
a side finding below). Batching: `tput --num-requests 2N --concurrency N --max-tokens 128`; pbatch:
`tput --num-requests N --concurrency N --max-tokens 1`; raw: 8 requests one at a time, 256 out.
`--warmup 2`, `--gpu-mem-util 0.90`, `--max-model-len 2048` (harness defaults) for both engines.
Wrappers: `scripts/bench.sh`, `scripts/sweep.sh`.

## 3. Reproduced baseline on current main

TBD

## 4. What changed since the measured commit (8fb7e90e6)

TBD

## 5. Attribution of the remaining batching gap

TBD

## 6. The fix

TBD

## 7. Before/after

TBD

## 8. Token equality

TBD

## 9. Cross-model, load and memory

TBD

## 10. Tests

TBD

## 11. Remaining gap

TBD

## 12. Side findings

TBD

## 13. Playbook notes

TBD
