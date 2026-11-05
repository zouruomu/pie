# status — 20261105-2033-h100-sxm-x1-gpt-oss-120b-batching

- **UTC:** 2026-11-05 20:55
- **Phase:** reproduce (round 1 batching sweep c32–c1024: vLLM → SGLang → pie, all on current main e4bb499c)
- **Last result:** smoke pie c32 = 2132 tok/s (pie-rendered 64-token prompts; the real runs pass `--pretokenized-prompts` so every engine gets the identical 128-token HF-rendered ids).
  - KV pool: pie declares 6192 MiB; vLLM 0.32.0 reports 7.68 GiB / 111,357 tokens.
  - pie reserves 2047 MiB for graph bodies, of which 139 MiB is used once all 135 are armed.
- **Leads (not yet measured):**
  - (1) gpt_oss.poem pages its sliding-window layers in the full KV space. Gemma 4 gives them a windowed ring space. Experiment engine building.
  - (2) the bodies reserve.
  - (3) #1368's admission cap.
- **Running:** sweep1 (vLLM), build of the windowed-space experiment.
- **Blocker:** none
- **ETA hand-over:** ~06:40Z
