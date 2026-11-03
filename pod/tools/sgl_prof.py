# Drive SGLang exactly as sglang_bench.py --report-timing does (same engine
# kwargs, streamed requests), but bracket ONE measured request with SGLang's
# own CUDA_PROFILER start/stop (cudaProfilerStart runs in the scheduler
# process, which is what nsys --capture-range=cudaProfilerApi needs).
import argparse, asyncio, sys, time
sys.path.insert(0, "/root/src/pie/scripts/bench")
import sglang_bench as sb
from common import add_mode_subcommands, hf_chat_prompts_and_counts, make_prompts

def main():
    parser = argparse.ArgumentParser()
    add_mode_subcommands(parser)
    args = parser.parse_args(["latency", "--model", "Qwen/Qwen3.5-0.8B", "--requests", "2",
                              "--warmup", "2", "--max-tokens", "256",
                              "--prompts-file", "/root/prompts-512.json"] + sys.argv[1:])
    prompts, counts = hf_chat_prompts_and_counts(args.model, args.system, make_prompts(args, 4), getattr(args, "think", None))
    engine = sb.build_engine(args, 1)
    params = {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": 256, "ignore_eos": args.ignore_eos}

    async def gen(p):
        t = time.perf_counter()
        n = 0
        g = await engine.async_generate(prompt=p, sampling_params=params, stream=True)
        async for out in g:
            n = out["meta_info"]["completion_tokens"]
        return n, time.perf_counter() - t

    for p in prompts[:2]:
        print("warm", engine.loop.run_until_complete(gen(p)), flush=True)
    engine.start_profile(activities=["CUDA_PROFILER"])
    print("measured", engine.loop.run_until_complete(gen(prompts[2])), flush=True)
    engine.stop_profile()
    print("after", engine.loop.run_until_complete(gen(prompts[3])), flush=True)
    engine.shutdown()

if __name__ == "__main__":
    main()
