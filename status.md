# Status: 20261103-2218-l4-x1-gpt-oss-20b-batching
Phase: 2 (baseline sweep on current main 48a36f0d8 running). Updated 2026-11-03 22:45Z. Deadline ~2026-11-04 14:20Z.

Done:
- Builds: pie engine (base-48a36f0d8, sha 07db7e20...), text-completion-bench wasm, vLLM 0.32.0 venv, gpt-oss-20b downloaded.
- Harness: site metrics decoded from db.js (see notes below); prompts p225/p609 made with HF chat-rendered exact counts.
- Smoke: pie pf8 on main 7.7k prefill tok/s (site at 8fb7e90e6: 4.67k); vLLM pf8 6.2k.

Running: results/base, 3 alternating rounds pie/vLLM over raw, pf4..pf1024, c4..c1024.
Next: analyse curves, profile where pie trails.
