# Status: 20261105-2035-b200-x1-minimax-m3-1-flash-decode

- **UTC:** 2026-11-05 21:05
- **Phase:** setup (reproduce)
- **Running:** miniature cut download (MiniMaxAI/MiniMax-M3.1-Flash @ce7f4df8, layers 0-59, 16 experts, 43.4 GiB; MXFP8 projections + NVFP4 experts), pie engine build, vLLM 0.32.0 and SGLang 0.5.22 installs.
- **Last result:** none yet.
- **Blocker:** none. anton-mel/stats has no MiniMax recipe on any branch and pie-project/benchmarks is not readable; the cut is reproduced from the brief's description (60/60 layers, 16 of 128 experts) with `scripts/bench/shrink_checkpoint.py`.
- **ETA:** early verdict ~23:15 UTC; hand-over ~06:45 UTC.
