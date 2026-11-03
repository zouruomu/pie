# r2g-control

Control branch for Felix's Pie red-to-green automation, which works through red cells on benchmarks.pie-project.org. The work itself lands in pull requests against pie-project/pie.

- `bootstrap.sh`: the start command of each run's Runpod pod.
- `pod/`: the pod agent's Claude Code settings, user instructions and commit hook.
- `runs/<run>/`: the orchestrator's brief, its reviews, and a `done` flag for each run.

Each run's results go to `r2g-evidence/<run>`, and its fix goes to `r2g/<run>`.
- `orchestrator-prompt.md`: the template for the routine's prompt. The routine embeds a copy with `__INFRA_SHA__` filled in, so editing this file changes nothing until the routine is updated.
- `ledger.md`: one entry per finished run. Orchestrators read it before choosing a target.
- `tools/red_cells.py`: lists the site's red cells, mirroring its own rule.

Infrastructure (`bootstrap.sh`, `pod/`, `tools/`) is pinned by commit in the routine, so a pod can't change what later runs execute. Run state (`runs/`, `ledger.md`) is read from the latest commit.
