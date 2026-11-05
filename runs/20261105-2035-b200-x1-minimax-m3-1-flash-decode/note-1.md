# Note 1

1. The site lists this miniature at 124.25 B parameters (23.0 B active, 2.458 B embedding, `mini_k` 0.125, 60 layers). Count your cut's parameters and put both numbers in report.md. If they disagree by more than rounding, say how (e.g. a different expert count) and whether it can matter for decode at concurrency 1 (top-4 routing reads 4 experts a layer either way). Don't spend more than 15 minutes on this; it isn't a blocker.
2. Your status.md says 21:05 UTC at 20:54 UTC by the commit time. Use `date -u` for the UTC line.
