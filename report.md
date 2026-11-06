# Report: B200 × MiniMax-M3.1-Flash (miniature) × decode

Draft, in progress. Anything not yet measured is marked pending.

## The cut and its parameters (note 1)

- Cut A: `MiniMaxAI/MiniMax-M3.1-Flash` @ce7f4df8 (weights unchanged since the 10-23 init), `scripts/bench/shrink_checkpoint.py --layers 0-59 --experts 16`: 43.41 GiB, 13 024 tensors, MXFP8 projections, NVFP4 experts kept as released. The site's row (`data/tags/b200-full.json`, pie 0d857cda6) states `base_model` MiniMaxAI/MiniMax-M3.1-Flash, `mini_layers` 60, `mini_k` 0.125, `mini_params_b` 124.25, `mini_active_b` 23.0, `mini_emb_b` 2.458.
- My count (`results/scripts/count_params.py`, NVFP4 planes counted two values a byte; `results/param_count.txt`): text weights 64.67 B (routed 51.64, attention/dense/router/norms 7.34, shared expert 3.23, embedding + lm_head 2.458), active 25.9 B (top-4 + shared + the rest); block scales 3.56 B and the vision tower 0.87 B apart.
- The embedding matches the site exactly. 124.25 B matches no integer expert count: with 56.6 M parameters per expert over 57 routed layers it would take about 35 of 128 experts, while `mini_k` 0.125 is 16 of 128, the cut I made. I read 124.25 / 23.0 as the site's own formula, not a different cut. For decode at concurrency 1 it can't matter: each routed layer reads top-4 + shared experts whatever the bank size.
- Cut B: `MiniMaxAI/MiniMax-M3-MXFP8` cut the same way: 65.95 GiB, same architecture (60 layers, 128 experts, top-4, hidden 6144), every expert MXFP8. Text weights 64.65 B, active 25.9 B (`results/param_count_m3mxfp8.txt`).

## Why two cuts

- vLLM 0.32.0, vLLM 0.33.0 and SGLang 0.5.22 cannot load cut A: each fails with "size of tensor a (192) must match the size of tensor b (384)" (`results/raw/vllm-r1.log`, `vllm033-r1.log`, `sglang-r1.log`). They read the NVFP4 experts' 1x16 scales (K/16 = 384) as MXFP8 1x32 scales (K/32 = 192). vLLM's recipe and SGLang's cookbook serve M3.1-Flash only from dev images (`vllm/vllm-openai:minimax-m31-flash`; `lmsysorg/sglang:dev-minimax-m31-flash`, "support not in main yet").
- So the site's vLLM 0.32.0 and SGLang 0.5.22 numbers were not measured on this checkpoint at those versions. Cut B is the same architecture in the form those versions do load ("the MXFP8 release": the site's scheme wording, and the comment on `minimax_m3_vl` in shrink_checkpoint.py), so it is the fair same-weights comparison I can reproduce.

## Side finding: pie serves both cuts as bf16 by default

- `crates/runtime/src/engine/load.rs:267` (`identify`) returns the first deployment whose contract compiles. For a miniature, `Catalog::candidates_of` tries the cut deployments in package order, and `models/minimax_m3.poem` lists `minimax-m3-mini`'s bf16 deployment first. So both cuts load as `minimax-m3-mini-bf16-kv-bf16-l60-e16`: the MXFP8/NVFP4 planes widen to bf16 (124.5 GB on the GPU) and the load needs more than this pod's 264 GiB cgroup on the host (OOM-killed twice, `results/logs/mem-r1b.log`). `--precision` can't override it: the engine refuses after identification ("stores its weights at [Bf16] and the config asks for [Mxfp8, Nvfp4]").
- For every measurement here pie serves the release precision, via the package order in `~/.pie/models/minimax_m3.poem` (the mini's [MX8, nvfp4a4] and MX8 deployments listed ahead of bf16, a local edit only). Each pie log records its deployment: `minimax-m3-mini-mxfp8a8g32-nvfp4a4-kv-bf16-l60-e16` (cut A), `minimax-m3-mini-mxfp8a8g32-kv-bf16-l60-e16` (cut B).

## Reproduced baseline (pending)
