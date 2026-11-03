# r2g-control

Control branch for Felix's Pie red-to-green automation, which works through red cells on benchmarks.pie-project.org. The work itself lands in pull requests against pie-project/pie.

- `bootstrap.sh`: the start command of each run's Runpod pod.
- `pod/`: the pod agent's Claude Code settings, user instructions and commit hook.
- `runs/<run>/`: the orchestrator's brief, its reviews, and a `done` flag for each run.

Each run's results go to `r2g-evidence/<run>`, and its fix goes to `r2g/<run>`.
