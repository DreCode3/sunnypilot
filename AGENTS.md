# Codex Operating Policy

## Remote Comma Update Standard (Required)
When operating on a comma device over unreliable remote SSH, all update/install actions must use a detached, idempotent, offroad-safe workflow.

Rules:
- Launch updates as a detached job on-device (`tmux new -d` preferred, `nohup` acceptable) so execution survives SSH drops.
- Gate any git/update/reboot action on `IsOffroad=true`; abort if onroad.
- Use a lock (`flock`) so only one update job can run at a time.
- Fetch only the target branch ref (`refs/heads/<branch>`) instead of broad fetches.
- Use fast-forward-only update behavior for branch sync.
- Sync/update submodules as part of the update job.
- Write full logs to `/data/media/0/update_logs/<timestamp>.log`.
- Write an explicit success/failure status file.
- Reboot only if update succeeded and device is still offroad.
- On reconnect, verify and report final branch + commit hash and log/status file paths.

