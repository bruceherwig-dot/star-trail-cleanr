#!/usr/bin/env bash
# Watch the latest Star Trail CleanR GitHub Actions build.
#
# Usage:
#   bash scripts/watch_ci.sh                  # watch until the build finishes
#   bash scripts/watch_ci.sh --max-minutes 5  # watch for 5 minutes, then stop
#
# Polls the latest workflow run every 45 seconds using the gh CLI (authenticated).
# Exits 0 on success, 1 on failure, 2 when the time limit ran out first.
#
# WHY THE TIME LIMIT EXISTS. A release build takes about 40 minutes, and Claude
# only speaks when something wakes it up. Watching straight through means going
# silent for the whole build and Bruce having to ask "update?" himself, which he
# did repeatedly on 2026-08-30. He asked for an update every five minutes and,
# offered a version that only spoke when a job changed state, said: "I don't
# trust you... every 5 minutes."
#
# So the release procedure runs this in FIVE MINUTE CHUNKS, in the background.
# Each chunk ending is what wakes Claude up: it reports, then starts the next
# chunk. The prompting comes from the process exiting, not from Claude
# remembering to check, which is the only kind of reminder that has ever worked
# on this project.
#
# Exit 2 means "not finished yet, start another chunk" -- it is NOT a failure.

MAX_MINUTES=0            # 0 = no limit, watch until the build finishes
while [ $# -gt 0 ]; do
  case "$1" in
    --max-minutes) MAX_MINUTES="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done
DEADLINE=0
if [ "$MAX_MINUTES" -gt 0 ] 2>/dev/null; then
  DEADLINE=$(( $(date +%s) + MAX_MINUTES * 60 ))
fi

REPO="bruceherwig-dot/star-trail-cleanr"

run_json=$(gh run list --repo "$REPO" --limit 1 --json databaseId,headBranch,status,conclusion,url 2>&1) \
  || { echo "gh CLI error: $run_json"; exit 1; }

RUN_ID=$(printf '%s' "$run_json" | python3 -c "import json,sys; print(json.load(sys.stdin)[0]['databaseId'])")
TAG=$(printf '%s' "$run_json"    | python3 -c "import json,sys; print(json.load(sys.stdin)[0]['headBranch'])")
URL=$(printf '%s' "$run_json"    | python3 -c "import json,sys; print(json.load(sys.stdin)[0]['url'])")

echo "Watching build for ${TAG} (run ${RUN_ID})"
echo "  ${URL}"
echo

while true; do
  if [ "$DEADLINE" -gt 0 ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then
    # Time limit reached with the build still running. Print the per-job picture
    # so the report Claude makes is about what is actually happening, not just
    # "still going", then exit 2 so the caller knows to start another chunk.
    echo
    echo "Still building after ${MAX_MINUTES} minute(s). Jobs so far:"
    gh run view "$RUN_ID" --repo "$REPO" --json jobs \
      --jq '.jobs[] | "  \(.conclusion // .status)  \(.name)"' 2>/dev/null \
      || echo "  (could not read the job list)"
    exit 2
  fi
  row=$(gh run view "$RUN_ID" --repo "$REPO" --json status,conclusion 2>&1) \
    || { echo "gh CLI error: $row"; sleep 45; continue; }
  status=$(printf '%s' "$row" | python3 -c "import json,sys; r=json.load(sys.stdin); print(r.get('status','unknown'))")
  conclusion=$(printf '%s' "$row" | python3 -c "import json,sys; r=json.load(sys.stdin); print(r.get('conclusion') or '-')")
  echo "$(date '+%H:%M:%S')  ${status} ${conclusion}"
  case "${status} ${conclusion}" in
    "in_progress -"|"queued -")
      sleep 45
      ;;
    "completed success")
      echo "Build succeeded."
      exit 0
      ;;
    completed*)
      echo "Build did not succeed: ${status} ${conclusion}"
      exit 1
      ;;
    *)
      sleep 45
      ;;
  esac
done
