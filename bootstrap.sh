#!/bin/bash
# Pie red-to-green pod bootstrap, run as the pod's start command.
#
# Starts the image's own services, sets up the pod agent (headless Claude Code) to
# work in Felix's name, runs it on the run's brief, answers the orchestrator's
# reviews, and terminates the pod when the orchestrator marks the run done or the
# deadline passes.
#
# Set by the orchestrator: R2G_RUN, R2G_CONTROL_SHA (the pinned commit this script, pod/ and the
# tools come from; briefs and reviews are read from the latest r2g-control), R2G_DEADLINE_H (default 16).
# Runpod secrets: claude_code_oauth_token, r2g_github_token.
#
# Branches on zouruomu/pie:
#   r2g-control             this script, pod config, briefs, reviews, done flags (orchestrator writes)
#   r2g/<run>               the fix (pod agent writes; the PR head)
#   r2g-evidence/<run>      reports, raw results, logs (pod writes)
set -u
R=/root/r2g
mkdir -p "$R/evidence"
exec > >(tee -a "$R/bootstrap.log") 2>&1
log() { echo "[$(date -u +%FT%TZ)] $*"; }

RUN=${R2G_RUN:?}
SHA=${R2G_CONTROL_SHA:?}
export CLAUDE_CODE_OAUTH_TOKEN=${RUNPOD_SECRET_claude_code_oauth_token:-}
export R2G_GH_TOKEN=${RUNPOD_SECRET_r2g_github_token:-}
export HF_TOKEN=${HF_TOKEN:-${RUNPOD_SECRET_HF_TOKEN:-}}
unset RUNPOD_SECRET_claude_code_oauth_token RUNPOD_SECRET_r2g_github_token
log "run $RUN, control $SHA, pod ${RUNPOD_POD_ID:-?}," \
    "claude token $([ -n "$CLAUDE_CODE_OAUTH_TOKEN" ] && echo set || echo MISSING)," \
    "github token $([ -n "$R2G_GH_TOKEN" ] && echo set || echo MISSING)"

# Tailscale SSH, sshd and pod-update, without the two tokens in their environment.
env -u CLAUDE_CODE_OAUTH_TOKEN -u R2G_GH_TOKEN /usr/local/bin/start.sh > "$R/start.log" 2>&1 &

gitcfg() {
    git -C "$1" config user.name zouruomu
    git -C "$1" config user.email zouruomu@gmail.com
    git -C "$1" config commit.gpgsign false
    git -C "$1" config credential.helper \
        '!f() { echo username=x-access-token; echo "password=$R2G_GH_TOKEN"; }; f'
}

git -C "$R/evidence" init -q
git -C "$R/evidence" checkout -q -b "r2g-evidence/$RUN"
git -C "$R/evidence" remote add origin https://github.com/zouruomu/pie
gitcfg "$R/evidence"

sync_evidence() { bash "$R/sync.sh"; }

terminate() {
    log "terminating the pod: $1"
    sync_evidence
    if command -v runpodctl > /dev/null && runpodctl remove pod "$RUNPOD_POD_ID"; then
        sleep infinity
    fi
    curl -sS -o /dev/null -w "REST delete: %{http_code}\n" -X DELETE \
        "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY:-}"
    log "self-termination may have failed; waiting for the orchestrator"
    sync_evidence
    sleep infinity
}

( sleep $(( ${R2G_DEADLINE_H:-16} * 3600 )); terminate "deadline of ${R2G_DEADLINE_H:-16} h" ) &

curl -fsSL "https://raw.githubusercontent.com/zouruomu/pie/$SHA/pod/sync.sh" -o "$R/sync.sh"
git clone -q --filter=blob:none https://github.com/zouruomu/pie "$R/pie" || terminate "clone failed"
gitcfg "$R/pie"
git -C "$R/pie" remote add upstream https://github.com/pie-project/pie
git -C "$R/pie" fetch -q upstream main
git -C "$R/pie" fetch -q origin r2g-control
git -C "$R/pie" show "$SHA:pod/commit-msg" > "$R/pie/.git/hooks/commit-msg"
chmod +x "$R/pie/.git/hooks/commit-msg"
git -C "$R/pie" checkout -q -B "r2g/$RUN" upstream/main

export CLAUDE_CONFIG_DIR="$R/claude-config" IS_SANDBOX=1 R2G_EVIDENCE="$R/evidence"
mkdir -p "$CLAUDE_CONFIG_DIR"
git -C "$R/pie" show "$SHA:pod/settings.json" > "$CLAUDE_CONFIG_DIR/settings.json"
git -C "$R/pie" show "$SHA:pod/CLAUDE.md" > "$CLAUDE_CONFIG_DIR/CLAUDE.md"
git -C "$R/pie" show "origin/r2g-control:runs/$RUN/brief.md" > "$R/brief.md" || terminate "no brief"
git -C "$R/pie" show "$SHA:pod/PLAYBOOK.md" > "$R/PLAYBOOK.md"
mkdir -p "$R/tools"
for f in $(git -C "$R/pie" ls-tree --name-only "$SHA" pod/tools/); do
    git -C "$R/pie" show "$SHA:$f" > "$R/tools/$(basename "$f")"
done

agent() {
    local i=$1
    shift
    log "agent run $i starting"
    (cd "$R/pie" && claude -p --model claude-opus-5-5 --dangerously-skip-permissions \
        --output-format json "$@") > "$R/agent-$i.json" 2> "$R/agent-$i.err"
    log "agent run $i exited with $?"
    tail -c 2000 "$R/agent-$i.err"
    sync_evidence
}

agent 1 "Read /root/r2g/PLAYBOOK.md first and follow it. Your brief:

$(cat "$R/brief.md")"
SID=$(grep -o '"session_id": *"[^"]*"' "$R/agent-1.json" | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
log "agent session ${SID:-unknown}"

# Waiting on the orchestrator: no review and no done for R2G_IDLE_H hours means it has
# gone away, so the pod stops; a later run picks the work up from the evidence branch.
k=1
idle_since=$(date +%s)
while true; do
    sleep 120
    if (( $(date +%s) - idle_since > ${R2G_IDLE_H:-4} * 3600 )); then
        terminate "no review or done for ${R2G_IDLE_H:-4} h"
    fi
    git -C "$R/pie" fetch -q origin r2g-control || continue
    if git -C "$R/pie" cat-file -e "origin/r2g-control:runs/$RUN/done" 2> /dev/null; then
        terminate "the orchestrator marked the run done"
    fi
    if git -C "$R/pie" cat-file -e "origin/r2g-control:runs/$RUN/review-$k.md" 2> /dev/null; then
        agent $((k + 1)) --resume "$SID" "$(git -C "$R/pie" show "origin/r2g-control:runs/$RUN/review-$k.md")"
        k=$((k + 1))
        idle_since=$(date +%s)
    fi
done
