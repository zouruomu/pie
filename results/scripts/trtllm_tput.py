# TRT-LLM (PyTorch backend, LLM API) throughput bench mirroring pie_bench/vllm_bench `tput`:
# the same prompts file and system prompt rendered by HF's chat template, `--num-requests 2c
# --concurrency c` (a semaphore of c), 128 output tokens (ignore_eos), greedy, KV block reuse off
# unless --reuse. Metrics as scripts/bench/common.py summarize(): output tok/s = output tokens /
# wall; TTFT and latency from each request's own start; ITL from per-token arrival gaps.
import argparse
import asyncio
import hashlib
import json
import statistics
import subprocess
import time

SYSTEM = "You are a helpful benchmarking assistant."


def clocks():
    q = "clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used"
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts-file", required=True)
    p.add_argument("--points", default="64,128,256,512,1024")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--requests-per-c", type=int, default=2)
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--max-num-tokens", type=int, default=None)
    p.add_argument("--max-batch-size", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--mem-frac", type=float, default=None)
    p.add_argument("--ssm-dtype", default=None, help="kv_cache_config.mamba_ssm_cache_dtype")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--json-out", required=True)
    p.add_argument("--dump-ids", default=None)
    a = p.parse_args()

    from transformers import AutoTokenizer
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig
    import tensorrt_llm
    import torch

    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = json.load(open(a.prompts_file))
    points = [int(x) for x in a.points.split(",")]
    need = a.warmup + a.requests_per_c * max(points)
    ids = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": u}],
                                   add_generation_prompt=True, tokenize=True)
           for u in prompts[:need]]
    ids = [list(x["input_ids"]) if not isinstance(x, list) else x for x in ids]

    kv = dict(enable_block_reuse=a.reuse)
    if a.mem_frac:
        kv["free_gpu_memory_fraction"] = a.mem_frac
    if a.ssm_dtype:
        kv["mamba_ssm_cache_dtype"] = a.ssm_dtype
    kw = dict(kv_cache_config=KvCacheConfig(**kv))
    if a.max_num_tokens:
        kw["max_num_tokens"] = a.max_num_tokens
    if a.max_seq_len:
        kw["max_seq_len"] = a.max_seq_len
    if a.max_batch_size:
        kw["max_batch_size"] = a.max_batch_size
    llm = LLM(model=a.model, **kw)
    sp = SamplingParams(max_tokens=a.max_tokens, temperature=0.0, ignore_eos=True)

    async def run(idx, conc):
        sem = asyncio.Semaphore(conc)
        res = {}

        async def one(i):
            async with sem:
                t0 = time.perf_counter()
                arr, n = [], 0
                async for o in llm.generate_async(ids[i], sp, streaming=True):
                    k = len(o.outputs[0].token_ids)
                    now = time.perf_counter()
                    arr += [now] * (k - n)
                    n = k
                    toks = list(o.outputs[0].token_ids)
                res[i] = (t0, arr, toks)

        t = time.perf_counter()
        await asyncio.gather(*(one(i) for i in idx))
        return time.perf_counter() - t, res

    loop = asyncio.new_event_loop()
    loop.run_until_complete(run(range(a.warmup), a.warmup))
    rows, dumps = [], {}
    for r in range(a.rounds):
        for c in points:
            idx = list(range(a.warmup, a.warmup + a.requests_per_c * c))
            ck0 = clocks()
            torch.cuda.nvtx.range_push(f"measured c{c} r{r}")
            wall, res = loop.run_until_complete(run(idx, c))
            torch.cuda.nvtx.range_pop()
            ck1 = clocks()
            ttft = [v[1][0] - v[0] for v in res.values()]
            lat = [v[1][-1] - v[0] for v in res.values()]
            gaps = [(b - x) * 1000 for v in res.values() for x, b in zip(v[1], v[1][1:])]
            out = sum(len(v[2]) for v in res.values())
            row = dict(point=f"c{c}", round=r, requests=len(idx), concurrency=c,
                       prompt_tokens=sum(len(ids[i]) for i in idx), output_tokens=out,
                       wall_s=wall, output_tok_per_s=out / wall,
                       ttft_p50_ms=pct(ttft, .5) * 1000, ttft_p99_ms=pct(ttft, .99) * 1000,
                       latency_p50_ms=pct(lat, .5) * 1000, latency_p99_ms=pct(lat, .99) * 1000,
                       intertoken_p50_ms=pct(gaps, .5), intertoken_p99_ms=pct(gaps, .99),
                       clocks_before=ck0, clocks_after=ck1)
            print(json.dumps(row), flush=True)
            rows.append(row)
            dumps[f"c{c}-r{r}"] = [res[i][2] for i in idx]
    out = dict(engine="tensorrt_llm", version=tensorrt_llm.__version__, model=a.model,
               config=dict(vars(a)), rows=rows,
               prompt_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    json.dump(out, open(a.json_out, "w"), indent=1)
    if a.dump_ids:
        json.dump(dumps, open(a.dump_ids, "w"))
    llm.shutdown()


if __name__ == "__main__":
    main()
