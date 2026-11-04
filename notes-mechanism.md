# Working notes: c8/c16 mechanism (folded into report.md §6)

A routed MXFP4 select with bf16 activations (gpt-oss's gate-up and down banks, 48 a decode step) picks
its kernel in `crates/kernels-cuda/src/linear/moe/select_fp4.rs`:

* `matmul_select_mxfp4` (`select_fp4.rs:829`) first asks `mxfp4_mma_worth` (`select_fp4.rs:99`), i.e.
  `mxfp4_mma_spans` (`select_fp4.rs:27-32`): `mxfp4_mma_from <= routes <= mxfp4_mma_until` **and**
  `ceil(routes / experts) >= mxfp4_mma_per_from`. On the L4: from 28 (`profiles/sm89.toml`), until
  16777216, per_from 4 (`profiles/sm89/nvidia-l4.toml:38`, copied from the L40S file).
* If that fails it groups only past `mxfp4_grouped_reuse` (2) routes an expert
  (`select_fp4.rs:905-907`), else runs the by-route kernel `moe_matmul_select_mxfp4`
  (`kernels/linear/moe/select.cu:434`), which streams the routed expert's codes once **per route**.

gpt-oss-20b has 32 experts, top-4. A decode of c lanes has 4c routes:

| lanes | routes | ceil(routes/32) | base kernel | fix kernel |
|---|---|---|---|---|
| 4 | 16 | 1 | below mma_from: by-route | same |
| 8 | 32 | 1 | by-route (32 ≤ 2×32) | run (`moe_select_mxfp4_mma_run`) |
| 16 | 64 | 2 | by-route (64 ≤ 2×32) | run |
| 32+ | 128+ | 4+ | run / staged | same |

At c16 the 64 routes touch ~28 distinct experts (32·(1-(31/32)^64)); the by-route kernel reads 64
expert slices, the run kernel each touched expert once: ~2.3x the bytes on a 300 GB/s card.
nsys (results/prof/prof-c16-*.kernels.txt): `moe_matmul_select_mxfp4` 977 calls 821.7 ms over 20.4
steps = 0.84 ms/call, 40.3 ms of a ~54 ms step; on the fix `moe_select_mxfp4_mma_run` 991 calls 463.8
ms = 0.47 ms/call, 22.5 ms a step. ITL p50 50.2 → 31.3 ms.

vLLM 0.32.0 on this card: `Using 'MARLIN' Mxfp4 MoE backend` (oracle
`vllm/model_executor/layers/fused_moe/oracle/mxfp4.py:614`), `fused_marlin_moe`
(`vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:224`) at every batch size: one
register-dequant kernel that reads each expert once a step, whatever the routes.
