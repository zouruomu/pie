# Status: 20261106-0130-b200-x1-minimax-m3-1-flash-decode

UTC: 2026-11-06 01:57Z
Phase: reproduce. The cell as the site states it does not reproduce as-is; sorting out which checkpoint each engine can load.

## Findings so far (raw logs in results/raw, results/logs)
1. Cut: MiniMaxAI/MiniMax-M3.1-Flash @ce7f4df8 (weights unchanged since 10-23), layers 0-59, 16 of 128 experts: 43.41 GiB, MXFP8 projections + NVFP4 experts. Site DB row (data/tags/b200-full.json) confirms mini_k 0.125, 60 layers, base_model MiniMaxAI/MiniMax-M3.1-Flash.
2. Params (note 1 / note-1 of the last run): my cut counts 64.67 B text params (routed 51.64, other 7.34, shared 3.23, embedding 2.458 B, exactly the site's), active 25.9 B; vision 0.87 B and scales 3.56 B apart (results/param_count.txt). Site: 124.25 B / 23.0 B active. 124.25 matches no integer expert count (it would need ~35 of 128); the embedding matches exactly and mini_k 0.125 = 16/128, so I read 124.25 as the site's formula, not a different cut. For c=1 decode it can't matter: top-4 + shared are read per layer either way.
3. **vLLM 0.32.0 and SGLang 0.5.22 cannot load this checkpoint** (both: "size of tensor a (192) must match b (384)": they read the NVFP4 experts' K/16 scales as MXFP8 K/32 scales). vLLM's own recipe says M3.1-Flash needs vLLM >= 0.33.0 (released 11-02, the day before the site's "0.32.0" vLLM run). So the site's vLLM/SGLang numbers were not measured on this checkpoint at those versions, or the versions are mislabelled.
4. **pie auto-identifies this cut as bf16** (minimax-m3-mini-bf16-kv-bf16-l60-e16): identify (crates/runtime/src/engine/load.rs:267) takes the first deployment that compiles, and models/minimax_m3.poem lists the mini's BF deployment before [MX8, nvfp4a4]. It widens to 124.5 GB on the GPU and OOMs this pod's 264 GiB cgroup on the host. --precision can't override it (refused after identify). With the mini's [MX8, nvfp4a4] listed first (an edit in ~/.pie/models only, to be reverted), pie serves it at 44 GB: **ITL p50 9.01 ms, 110 tok/s** (1 round), matching the site's 8.99.

## Running
- vLLM 0.33.0 install (own venv) -> vLLM on the M3.1-Flash cut.
- Cut of MiniMaxAI/MiniMax-M3-MXFP8 (same architecture, all-MXFP8 experts: "the MXFP8 release", the site's scheme wording and what 0.32.0/0.5.22 can load), 60 layers, 16 experts -> all three engines on it.

Blocker: none, but the cell's baseline is ambiguous; I will report both cuts.
ETA early verdict: ~03:30Z. Hand-over target: ~11:15Z.
Orchestrator note 1: acknowledged. Disk after the first cut: df -h / = 95G used / 156G free of 250G; one copy of each cut, shared by all engines.
