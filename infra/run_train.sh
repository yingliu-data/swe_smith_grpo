#!/usr/bin/env bash
# Supervisor loop for the trainer. Exits code 42 means "watchdog detected a
# stall" — retry. Any other non-zero exit is a real error: propagate. OOMs
# typically surface as SIGKILL (137); we retry those up to MAX_RETRIES then
# bail so the pod doesn't thrash.

set -euo pipefail

MAX_RETRIES="${MAX_RETRIES:-5}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-30}"

attempt=1
while [[ "$attempt" -le "$MAX_RETRIES" ]]; do
    echo "[supervisor] attempt $attempt / $MAX_RETRIES"
    set +e
    uv run train --resume latest "$@"
    rc=$?
    set -e

    case "$rc" in
        0)
            echo "[supervisor] trainer exited cleanly"
            exit 0
            ;;
        42)
            echo "[supervisor] watchdog stall detected (rc=42); retrying after ${COOLDOWN_SECONDS}s"
            ;;
        137)
            echo "[supervisor] OOM-kill detected (rc=137); retrying after ${COOLDOWN_SECONDS}s"
            ;;
        *)
            echo "[supervisor] unexpected exit code $rc; refusing to retry" >&2
            exit "$rc"
            ;;
    esac

    sleep "$COOLDOWN_SECONDS"
    attempt=$((attempt + 1))
done

echo "[supervisor] exhausted retries ($MAX_RETRIES); bailing" >&2
exit 1
