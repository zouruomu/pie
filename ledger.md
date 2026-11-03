# Red-to-green ledger

One entry per finished run, newest last. Orchestrators read it before choosing a target and append to it when they finish.

Format: `- <date> | <run> | <cells> | <outcome> | <PR> | notes for future runs`

- 2026-11-02 | pilot (manual) | rtx4090-x1 × qwen3.5-0.8b × decode (−11.3% vs SGLang) | fixed in part | pie-project/pie#1356 (draft) | Cause: the 4090 profile lacked `[compiler] dag_streams`, so the GDN b,a projection ran serially. The fix gives +1.8% (456 → 464 tok/s, SGLang 481). The remaining ~3.4% is kernel and compiler work: the matmul_split_rows copy, slow conv1d_update/gdn_step kernels, generated add+RMSNorm about 2× flashinfer's, the step epilogue, the attention prologue and merge. Pick this cell again only for that remaining work, after #1356 merges. Side finding: since #1338, Qwen3.5-4B bf16 no longer fits the 4090 at a 2048 context (reported to Anton).
