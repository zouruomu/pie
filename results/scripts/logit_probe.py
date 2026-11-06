# Teacher-forced logits through pie's embedded engine: the logit-probe inferlet (results/scripts/logit-probe/)
# fed request j's prompt (prompts[warmup + j], as pie_bench sends it) and then `steps` tokens of a reference
# greedy run's output; writes <out>/r<j>.f32, the logits of positions 0..steps (0: after the prompt) back to back.
# usage: logit_probe.py <model> <prompts.json> <teacher bench json> <out dir> <requests> <steps>
import asyncio, base64, json, os, sys
from pathlib import Path
sys.path.insert(0, "/root/r2g/pie/scripts/bench")
import pie_bench
from pie_client import Event

model, prompts_file, teacher_json, out, nreq, steps = sys.argv[1:7]
nreq, steps = int(nreq), int(steps)
WASM = Path("/root/probe/logit-probe/target/wasm32-wasip3/release/logit_probe.wasm")
prompts = json.load(open(prompts_file))
teacher = [r["output_token_ids"] for r in json.load(open(teacher_json))["requests"]]
os.makedirs(out, exist_ok=True)

async def main():
    args = pie_bench.build_parser().parse_args(["latency", "--model", model, "--requests", "1", "--warmup", "0"])
    async with pie_bench.python_pie_client(args) as (client, _):
        pkg = await client.install_program(WASM, force_overwrite=True)
        for j in range(nreq):
            proc = await client.launch_process(pkg, input={"prompt": prompts[2 + j], "teacher": teacher[j][:steps]})
            got = bytearray()
            while True:
                ev, msg = await proc.recv()
                if ev == Event.Message:
                    got += base64.b64decode(str(msg))
                elif ev == Event.Return:
                    Path(out, f"r{j}.f32").write_bytes(got)
                    print(j, msg, flush=True)
                    break
                else:
                    raise SystemExit(f"request {j}: {ev} {msg}")

asyncio.run(main())
