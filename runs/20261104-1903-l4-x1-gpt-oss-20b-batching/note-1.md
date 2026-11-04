# Note 1

1. Acknowledge this note in status.md. It is part of the infrastructure check.
2. One small extra, after both moe_select runs and only if it takes under 20 minutes. In the known failure, every output equals 3.3895e38, the test's 0x7f7f sentinel (moe_select.rs:113). So the pipe kernel wrote nothing, and its 3 × 33,792 B of shared memory is exactly sm_89's 99 KiB opt-in per block. Rerun just that test on the fix build with `CUDA_LAUNCH_BLOCKING=1` under `compute-sanitizer`. Record whether a launch or CUDA error appears, and what it says, in report.md. Don't fix it: it goes to Anton.
