# Side finding: a 60-layer, 16-expert cut of MiniMax-M3.1-Flash (or M3-MXFP8) is served as bf16

Kept apart from the perf fix, as note 2 asks. Not fixed in the PR branch.

## What happens

- `crates/runtime/src/engine/load.rs:267` (`identify`) walks `catalog().fits(..)` and returns the first deployment whose contract reads the checkpoint and whose storage plan compiles (`load.rs:292-293`). Nothing there prefers a deployment that reads the planes as stored over one that widens them.
- The order comes from `crates/poem-compiler/src/catalog.rs:367` (`Catalog::candidates`): non-miniatures first, then by `deploy.parts` (vision, drafter), then package order. `candidates_of` (`catalog.rs:382`) appends the miniature deployments cut to the checkpoint's depth and expert count after every listed one.
- No listed deployment fits a 60-layer, 16-expert cut (the full model's routers are 128 wide: "declares shape [128, 6144] but its expression yields [16, 6144]"), so the first cut deployment that compiles wins. `models/minimax_m3.poem:31-34` lists the miniature's bf16 deployment first (BF, U4, MX8, [MX8, nvfp4a4]), while the full model lists MX8 first (lines 26-30).
- So the MXFP8 + NVFP4 checkpoint serves as `minimax-m3-mini-bf16-kv-bf16-l60-e16`: every plane widened to bf16, 124.5 GB on the GPU against 44 GB as stored. On this pod the host side of that load passes the container's 264 GiB cgroup limit and is OOM-killed (twice: `results/raw/pie-base-r1.log`, `results/raw/pie-base-r1b.log`; memory trace `results/logs/mem-r1b.log`: 240 GB at 01:46:01, then killed).
- `[model] weight_precision` can't steer it: `load.rs:146` refuses after identification ("`minimax-m3-mini-bf16-kv-bf16-l60-e16` stores its weights at [Bf16] and the config asks for [Mxfp8 ..., Nvfp4 ...]; import the checkpoint at that precision to serve it"; `results/raw/probe-mx8nv.log`).
- `Catalog::identify` (`catalog.rs:406`) has a guard (`requantizes`, `catalog.rs:648`), but it only refuses a quantized plane re-quantized from another quantized form, and the runtime's `identify` doesn't call it.

## Minimal repro (B200, main e2e9d1e)

```
python scripts/bench/shrink_checkpoint.py --repo MiniMaxAI/MiniMax-M3.1-Flash --layers 0-59 --experts 16 \
    --out ~/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/mini60-e16
RUST_LOG=runtime::engine::load=info python scripts/bench/pie_bench.py latency \
    --model ~/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/mini60-e16 --requests 1
# -> serving deployment="minimax-m3-mini-bf16-kv-bf16-l60-e16" ... then the bf16 widening (OOM where host RAM < ~250 GB)
```

## How this run served the release precision

A local edit to the seeded package `~/.pie/models/minimax_m3.poem` (pie keeps a user's edits to it), listing the miniature's [MX8, nvfp4a4] and MX8 deployments ahead of BF. Every pie log records its deployment: `minimax-m3-mini-mxfp8a8g32-nvfp4a4-kv-bf16-l60-e16` on cut A, `minimax-m3-mini-mxfp8a8g32-kv-bf16-l60-e16` on cut B. The site's pie number (8.99 ms) matches the release-precision deployment here (8.97-9.01 ms), so the eval box evidently doesn't hit this, or it imports with an explicit `--sku`.

## Candidate fixes (not measured)

1. Package order: list `minimax-m3-mini`'s MX8 and [MX8, nvfp4a4] deployments ahead of BF, as the full model's are. One line moved; untested here whether a bf16 miniature checkpoint would then be claimed by MX8 (quantized at load), so it needs a bf16 miniature to check.
2. In `identify`, prefer a deployment whose contract reads every plane as stored over one that widens a quantized plane to bf16 (the counterpart of `requantizes`). More general; touches every model's identification.
