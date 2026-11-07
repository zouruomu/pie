# Outcome: 20261106-1706-h100-sxm-x1-qwen3-6-27b-batching

**Abandoned (written by the 2026-11-07 01:05Z orchestrator in housekeeping).** Not READY FOR REVIEW; no fix branch pushed.

## What happened
- Pod rcpud2it3ah8qm (H100 SXM, US-MO-1, $3.99/h) started 17:07Z on 11-06.
- The pod agent's run 1 ended at 20:16Z with exit 1: `api_error_status 401`, "Failed to authenticate: OAuth token revoked" (agent-1.json on the evidence branch). The bootstrap does not restart a failed agent, so the pod sat in its review wait.
- The owning orchestrator's last heartbeat was 20:20Z; no review or done followed, so the pod terminated itself at 00:17Z ("no review or done for 4 h"). Pod life ~7.2 h, about $28.6.
- The pod's background queues kept running after the agent died and pushed TRT-LLM best-config results at 00:17Z.

## What the evidence shows (my recomputation from results/ on r2g-evidence/<run>, 1 round each, site-style prompts ~158/196 tokens, 128 out)
| tok/s | c64 | c128 | c256 | c512 | c1024 |
|---|---|---|---|---|---|
| Qwen3.6 pie main dc6282d7 | 1990 | 2207 | **2211** | 2092 | 1816 |
| Qwen3.6 vLLM 0.32.0 | 1791 | 2151 | 2182 | – | – |
| Qwen3.6 TRT-LLM 1.4.0rc2.post1 (max_batch_size 128, max_seq_len 2048, reuse off) | 1978 | 2852 | 2914 | 2909 | **2916** |
| Qwen3.8 pie main | 1890 | 2090 | **2097** | 1994 | 1734 |
| Qwen3.8 vLLM 0.32.0 | 1699 | 2045 | 2063 | – | – |
| Qwen3.8 TRT-LLM | 1887 | 2651 | 2691 | 2703 | **2703** |

Best to best: Qwen3.6 −24.2%, Qwen3.8 −22.4% vs TRT-LLM; pie ≈ vLLM. No regression since the site's commit.

Cause (the agent's attribution, consistent with its experiments): for hybrid (GDN) trunks pie's load fit (`Pools::fit_the_card`, crates/engine-cuda/src/store.rs:1130-1163; `balanced_state_slots`, crates/engine/src/fit.rs:152) cuts recurrent-state slots until the kv pages hold every slot at the declared context (2048), while these sequences use ~290 tokens. pie seats ~72-92 lanes a fire against TRT-LLM's 128 at the same ~30 ms ITL. Experiments: `--max-model-len 512` gives c128 2751 (+25%); a slot-multiplier experiment engine (results/exp-seats) gives c128 x5 2782, x8 1911 (over-seating collapses), c512 x3 2444. Unshipped; needs a principled fix (size slots by expected use, admit by each request's own prompt+max_tokens, as TRT-LLM's GUARANTEED_NO_EVICT and vLLM's shared block pool do) plus tail measurements.
