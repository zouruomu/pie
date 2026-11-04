#!/bin/bash
# The pod agent's check-in: pushes the evidence branch, then prints the orchestrator's new notes and
# what has landed on upstream main since this branch's base. Traces stay on the pod (nsys records
# the environment), and a file holding a secret's value or a token-shaped string stops the push.
R=/root/r2g
E=$R/evidence
P=$R/pie
exec 9> "$R/sync.lock"
flock 9
cp "$R/bootstrap.log" "$R"/agent-*.json "$E/" 2> /dev/null
printf '%s\n' '*.nsys-rep' '*.qdrep' '*.qdstrm' '*.sqlite' > "$E/.git/info/exclude"
git -C "$E" add -A
token='sk-ant-[A-Za-z0-9_-]{30,}|github_pat_[A-Za-z0-9_]{50,}|gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}|rpa_[A-Za-z0-9]{30,}|tskey-[A-Za-z0-9-]{20,}'
staged() { (cd "$E" && git diff --cached --name-only --diff-filter=d -z | xargs -0r grep -l "$@" -- 2> /dev/null); }
hits=$( (staged -Ff /root/.r2g-secrets; staged -E "$token") | sort -u)
if [ -n "$hits" ]; then
    git -C "$E" reset -q
    echo "NOT PUSHED: these files hold a secret or a token-shaped string. Remove it and sync again."
    echo "$hits"
else
    git -C "$E" commit -q -m "r2g ${R2G_RUN:-run} evidence" || true
    git -C "$E" push -q -f origin HEAD || echo "evidence push failed"
fi
[ -n "$(find "$E/status.md" -mmin +60 2> /dev/null)" ] && echo "status.md is over an hour old: update it."

git -C "$P" fetch -q origin r2g-control
for n in $(git -C "$P" ls-tree --name-only "origin/r2g-control:runs/$R2G_RUN/" 2> /dev/null | grep '^note-'); do
    [ -e "$R/seen-$n" ] && continue
    echo "New orchestrator $n (acknowledge it in status.md):"
    git -C "$P" show "origin/r2g-control:runs/$R2G_RUN/$n"
    touch "$R/seen-$n"
done
git -C "$P" fetch -q upstream main
echo "upstream main: $(git -C "$P" rev-list --count HEAD..upstream/main) commits not on this branch"
files=$(git -C "$P" diff --name-only upstream/main...HEAD)
[ -n "$files" ] && git -C "$P" log --oneline HEAD..upstream/main -- $files
exit 0
