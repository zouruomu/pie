# Outcome: 20261108-1706-h200-x1-mimo-v2-6-flash-rl-batching

**Blocked before it started: the pod agent's Claude OAuth token is still revoked.** Pod 34xs0q0cy8dcn4 (H200 SXM, EU-FR-1, $5.29/h) was created at 17:08Z. The bootstrap ran (claude, github and hf tokens set), and agent run 1 exited 4 s after it started (17:10:26Z → 17:10:30Z): "Failed to authenticate: OAuth token revoked" (api_error_status 401; agent-1.json on r2g-evidence/<run>). This is the fifth failure in a row (20:16Z on 11-06; 01:09Z and 09:09Z on 11-07; 01:08Z and 17:10Z on 11-08). The Runpod secret `claude_code_oauth_token` still shows createdAt 2026-11-03T20:57Z, updatedAt null.

This run was also a deliberate probe: 8 h after the 09:00Z skip, still with no sign on r2g-control of a re-issue, one cheap try was the only way to tell. The answer is no.

Closed by the orchestrator at ~17:14Z. No work was done on the pod. Pod cost ~$0.6.

The brief is fresh and reusable as it is (checked against main 9d134db at 17:10Z): H200 × MiMo-V2.6-Flash-RL (48 layers, 16 of 256 experts) batching −50.5% vs vLLM (pie seats ~140 lanes vs ~660; TTFT p50 93 s), prefill −30.5%. Leading hypotheses in the brief: pie's 256-wide padded heads (keys 192, values 128) in the kv pages, and how admission/fit charges the 39 windowed layers.
