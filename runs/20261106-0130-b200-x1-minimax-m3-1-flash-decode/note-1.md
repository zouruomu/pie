# Note 1 (orchestrator, 2026-11-06 01:25Z)

Thanks for the early check-in.

1. Disk: the container disk is 250 GB and can't grow without restarting the pod. If your cut is ~125 GB, keep exactly one copy on disk and point all three engines at it (no per-engine conversions or HF cache duplicates), delete download caches as you go, and put `df -h /` in status.md after the download. If it can't fit with the builds, say so in status.md at once rather than working around it.
2. The earlier attempt's agent cut 16 of 128 experts to 43.4 GiB; the site says 124.25 B parameters. Settle which cut the harness uses before downloading 125 GB (the 15-minute check in the brief), and record the reasoning.
