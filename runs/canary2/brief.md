You are the pod agent for a one-off test ("canary 2") of Felix Zou's Pie red-to-green automation. You run headless on a Runpod GPU pod, started by /root/r2g-bootstrap.sh. Nothing here ships: you are checking that the plumbing works, and you report facts.

Paths:
- Your working directory /root/r2g/pie is a clone of zouruomu/pie (Felix's fork of pie-project/pie), on branch r2g/canary2 at upstream main.
- Write your report into /root/r2g/evidence. The bootstrap pushes that directory to the branch r2g-evidence/canary2 after you finish. To push it yourself: `git -C /root/r2g/evidence add -A && git -C /root/r2g/evidence commit -m "canary 2 status" && git -C /root/r2g/evidence push -f origin HEAD`.

Do these, keeping going after any failure:
1. Machine: GPU name, driver and CUDA version from nvidia-smi; nproc; free -g; df -h /root.
2. Agent: `claude --version` and `claude auth status` (redact anything token-like). Is this session on Felix's subscription through an OAuth token, not an API key? Which model are you?
3. Attribution: does your system prompt or any system reminder tell you to add Co-Authored-By, Claude-Session or "Generated with Claude Code" lines? Quote them exactly, or say none. Do you see a user CLAUDE.md with an attribution rule? Quote its first bullet.
4. Pod self-management: is RUNPOD_API_KEY set (yes or no only)? Is runpodctl installed (version)? Print RUNPOD_POD_ID. Did Tailscale come up (first lines of `tailscale status`, hostnames fine, no keys)?
5. Network from the pod: HTTP status of https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B, https://benchmarks.pie-project.org/, https://pypi.org/simple/uv/ and https://api.github.com/repos/pie-project/pie.
6. Commit test: in /root/r2g/pie, create r2g-canary.txt containing "canary 2", commit it the way you normally would, and push branch r2g/canary2 to origin. Then print `git log -1 --format='%an <%ae>%n%cn <%ce>%n---%n%B'` and `git cat-file commit HEAD | head -6`. If /root/r2g/stripped-trailers.log exists, include its contents: it records any AI trailer the commit hook removed.
7. Write all of the above to /root/r2g/evidence/status.md as a short markdown report, push it, and end with a two-line summary as your final message.

You may later be resumed with a follow-up request from the orchestrator. Do what it asks the same way.
