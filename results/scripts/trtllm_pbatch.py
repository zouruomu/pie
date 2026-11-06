# TRT-LLM (PyTorch backend, LLM API) prefill-batch / throughput bench mirroring pie_bench tput:
# the same prompts file (warmup prompts first), the same system prompt and chat template,
# every request offered at t=0 (concurrency = N), prefill tok/s = prompt tokens / wall.
# KV block reuse is off unless --reuse (cold prompts, as pie and vLLM run them).
import argparse
import asyncio
import hashlib
import json
import statistics
import subprocess
import time

SYSTEM = "You are a helpful benchmarking assistant."


def clocks():
    q = "clocks.sm,clocks.mem,power.draw,temperature.gpu"
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts-file", required=True)
    p.add_argument("--points", default="4,8,16,32,64,128,256,512,1024")
    p.add_argument("--warmup", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=1)
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--max-num-tokens", type=int, default=None)
    p.add_argument("--max-batch-size", type=int, default=None)
    p.add_argument("--moe-backend", default=None)
    p.add_argument("--mem-frac", type=float, default=None)
    p.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
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
    need = a.warmup + max(points)
    ids = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": u}],
                                   add_generation_prompt=True, tokenize=True)
           for u in prompts[:need]]
    ids = [list(x["input_ids"]) if not isinstance(x, list) else x for x in ids]

    kv = dict(enable_block_reuse=a.reuse)
    if a.mem_frac:
        kv["free_gpu_memory_fraction"] = a.mem_frac
    kw = dict(kv_cache_config=KvCacheConfig(**kv))
    if a.max_num_tokens:
        kw["max_num_tokens"] = a.max_num_tokens
    kw["max_batch_size"] = a.max_batch_size or max(points)
    if a.moe_backend:
        from tensorrt_llm.llmapi import MoeConfig
        kw["moe_config"] = MoeConfig(backend=a.moe_backend)
    if not a.cuda_graphs:
        kw["cuda_graph_config"] = None
    llm = LLM(model=a.model, **kw)
    sp = SamplingParams(max_tokens=a.max_tokens, temperature=0.0, ignore_eos=True)

    async def run(idx):
        t0 = time.perf_counter()
        ttft, out = {}, {}

        async def one(i):
            async for o in llm.generate_async(ids[i], sp, streaming=True):
                if i not in ttft and o.outputs[0].token_ids:
                    ttft[i] = time.perf_counter() - t0
                out[i] = list(o.outputs[0].token_ids)

        await asyncio.gather(*(one(i) for i in idx))
        return time.perf_counter() - t0, ttft, out

    loop = asyncio.new_event_loop()
    loop.run_until_complete(run(range(a.warmup)))
    rows, dumps = [], {}
    for r in range(a.rounds):
        for n in points:
            idx = list(range(a.warmup, a.warmup + n))
            ck0 = clocks()
            torch.cuda.nvtx.range_push(f"measured pf{n} r{r}")
            wall, ttft, out = loop.run_until_complete(run(idx))
            torch.cuda.nvtx.range_pop()
            ck1 = clocks()
            ptoks = sum(len(ids[i]) for i in idx)
            t = sorted(ttft.values())
            row = dict(point=f"pf{n}", round=r, requests=n, prompt_tokens=ptoks, wall_s=wall,
                       prefill_tok_s=ptoks / wall,
                       output_tokens=sum(len(v) for v in out.values()),
                       ttft_p50_ms=statistics.median(t) * 1000, ttft_max_ms=t[-1] * 1000,
                       clocks_before=ck0, clocks_after=ck1)
            print(json.dumps(row), flush=True)
            rows.append(row)
            dumps[f"pf{n}-r{r}"] = [out[i] for i in idx]
    res = dict(engine="tensorrt_llm", version=tensorrt_llm.__version__, model=a.model,
               config=dict(vars(a)), rows=rows,
               prompt_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    json.dump(res, open(a.json_out, "w"), indent=1)
    if a.dump_ids:
        json.dump(dumps, open(a.dump_ids, "w"))
    llm.shutdown()


if __name__ == "__main__":
    main()
