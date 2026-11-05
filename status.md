# status — 20261105-2033-h100-sxm-x1-gpt-oss-120b-batching

- **UTC:** 2026-11-05 21:33
- **Phase:** reproduce, round 2 (site-like flags)

## Early verdict (provisional, round 1 of 3, main e4bb499c, H100 SM 1980 MHz)

Batching: on my setup the site's gap does **not** reproduce. pie leads vLLM 0.32.0 from c64 up.

Setup: 128-token prompts as identical HF-rendered ids (`--pretokenized-prompts` for pie), 128 out, max(256, 2c) requests, no `--report-timing`.

| c | pie | vLLM | pie vs vLLM |
|---|---|---|---|
| 32 | 1933 | 1989 | −2.8% |
| 64 | 3112 | 3036 | +2.5% |
| 128 | 5031 | 3818 | +31.8% |
| 256 | **7437** | 4487 | +65.7% |
| 512 | 6511 | **5197** | +25.3% |
| 1024 | 5795 | 5030 | +15.2% |

- **Best point against best point:** pie 7437 (c256) vs vLLM 5197 (c512), +43%. The site has pie 4560 vs vLLM 7198.
- **Single round:** short runs (6–50 s), so no spread yet.
- **Suspected site differences:**
  - The site reports TTFT/ITL, so it likely ran `--report-timing`. That changes pie's inferlet: the prefill is sent alone and decode staged only after the first token.
  - It likely used more requests a point (vLLM TTFT p50 14.6 s means a deep queue).
  - Round 2 runs with those flags (4c requests, `--report-timing`).
- **pie's own chat template:** it renders the bench prompts as 64 tokens, against 128 with the HF template. All my pie runs use the HF ids.

**KV pool:**
- pie declares 5.5–6.5 GiB depending on c (6192 MiB at c32–c256, 5472 MiB at c512, 6480 MiB at c1024).
- vLLM has 7.68 GiB (111k tokens) at c32 and 96,997 tokens at c1024.
- pie's mean active lanes: 200 at c256, 171 at c512, 231 at c1024.

**SGLang 0.5.22:** OOM while swizzling MXFP4 weights at load (7.5 GiB reserved-unallocated). Retrying with `expandable_segments` in round 2.

- **Running:** windowed-KV experiment (gpt_oss sliding layers in a windowed kv space, as Gemma 4), then round 2.
- **Next:** raw prefill and pf32–pf256 (SGLang is the best engine there).
- **Blocker:** none
- **ETA hand-over:** ~06:40Z
