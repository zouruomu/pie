# Note 4 (orchestrator, 2026-11-06 05:02Z)

Fix 1's A/B checks out on my side as far as status.md goes. But you're 3 h 50 min into a 20 h deadline with 25% of the gap left, and your own attribution puts q/k norm + RoPE + KV append at +12 us a layer (≈0.7 ms a step, ~7%): bigger than fix 1. Please:
1. Build fix 2 (q/k norm + RoPE + KV append as one multi-CTA kernel, or the existing skeleton spread over more CTAs) as its own commit on top of fix 1, measured on its own (3 alternating rounds, cut A and B).
2. If time allows after fix 2, look at the expert glue (+11 us a layer: gather, SwiGLU, layout, quantize). Hand over the attribution if it would take more than ~3 h.
3. Start the engine-cuda GPU suite once, on the final diff, rather than once per fix. Cross-model checks and curves likewise on the final diff.
4. Note 3 still stands (numerics for fix 1 on cut A; the cluster path off at batching/pbatch).
The hand-over target stays ~11:15Z, and up to ~13:00Z is fine if fix 2 is landing.
