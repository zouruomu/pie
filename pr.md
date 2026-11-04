# Decode throughput part of the pie vs vLLM gap on L4 (gpt-oss c8/c16 selects on the run kernel)

gpt-oss-20b's c8 and c16 decodes route 32 and 64 routes over 32 experts: 1 and 2 an expert, under
the L4 profile's `moe.mxfp4_mma_per_from = 4`, which the file copied from the L40S's. Those
selects missed the register-dequant kernels. Not past the 2-an-expert grouping threshold either, they
ran the by-route `moe_matmul_select_mxfp4`, which streams an expert's codes once for each route: 0.84
ms a select, 40 ms of a c16 step (nsys), where vLLM runs Marlin at every width. With `= 0`, as the
A40 has, they take the 512-code run kernel that c25-c128 already decode on, reading each touched
expert once (0.47 ms a select); c17-c24 move to it off the grouped wmma kernel too. On an L4, 3
alternating rounds over the site's whole curve: c16 299 -> 500 tok/s (vLLM 0.32.0 442), c8 239 ->
276 (vLLM 276), c4 168 -> 174; every other point c32-c1024 and pf4-pf1024 within ±1.2% (pf8 -2.3%,
tracking the power-capped card's SM clock), raw decode 61.1 -> 61.0. After the rebase: {{FINAL_PR}}.
gptoss-20b-mini (16 experts) c8 794 -> 888, a 32-expert MiMo-V2.6-Flash miniature c8 463 -> 637 and
c4 309 -> 331, nothing slower. Greedy tokens are identical for raw and pbatch on all three models;
at c8/c16 the kernel changes on every decode step and outputs differ as much as base's own runs do
from each other. The engine-cuda GPU suite, the profile tests and kernels-cuda's routed-select tests
pass on both builds: {{TESTS_PR}}.
