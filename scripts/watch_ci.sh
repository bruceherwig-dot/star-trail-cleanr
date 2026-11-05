#!/usr/bin/env bash
# Watch a Star Trail CleanR GitHub Actions BUILD until it finishes.
#
# Usage:
#   bash scripts/watch_ci.sh                      # the newest run of the build workflow
#   bash scripts/watch_ci.sh 41918437171          # a specific run (use this after a rerun)
#   bash scripts/watch_ci.sh --max-minutes 5      # stop watching after 5 minutes (a "chunk")
#   bash scripts/watch_ci.sh 41918437171 --max-minutes 5
#
# Only runs of build.yml are considered. The old version took the newest run of ANY workflow, and a
# routine "pages build and deployment" run (which the release's own feed job triggers) became newest,
# so it reported "Build succeeded" for the wrong run (2026-11-04).
#
# Polls every 45 seconds using the gh CLI (authenticated) and prints each job's state.
# Exit codes: 0 = build succeeded, 1 = build failed or gh error, 2 = still running when --max-minutes ran out.

REPO="bruceherwig-dot/star-trail-cleanr"
WORKFLOW="build.yml"
RUN_ID=""
MAX_MINUTES=0

while [ $# -gt 0 ]; do
  case "$1" in
    --max-minutes) MAX_MINUTES="${2:-0}"; shift 2 ;;
    --max-minutes=*) MAX_MINUTES="${1#*=}"; shift ;;
    ''|*[!0-9]*) echo "Unknown argument: $1 (expected a run number or --max-minutes N)"; exit 1 ;;
    *) RUN_ID="$1"; shift ;;
  esac
done

if [ -z "$RUN_ID" ]; then
  run_json=$(gh run list --repo "$REPO" --workflow "$WORKFLOW" --limit 1 --json databaseId 2>&1) \
    || { echo "gh CLI error: $run_json"; exit 1; }
  RUN_ID=$(printf '%s' "$run_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[0]['databaseId'] if d else '')")
  [ -n "$RUN_ID" ] || { echo "No runs of ${WORKFLOW} found."; exit 1; }
fi

info=$(gh run view "$RUN_ID" --repo "$REPO" --json headBranch,headSha,url 2>&1) \
  || { echo "gh CLI error: $info"; exit 1; }
TAG=$(printf '%s' "$info" | python3 -c "import json,sys; print(json.load(sys.stdin)['headBranch'])")
SHA=$(printf '%s' "$info" | python3 -c "import json,sys; print(json.load(sys.stdin)['headSha'][:7])")
URL=$(printf '%s' "$info" | python3 -c "import json,sys; print(json.load(sys.stdin)['url'])")

echo "Watching build for ${TAG} at ${SHA} (run ${RUN_ID})"
echo "  ${URL}"
echo

START=$(date +%s)
while true; do
  row=$(gh run view "$RUN_ID" --repo "$REPO" --json status,conclusion,jobs 2>&1) \
    || { echo "gh CLI error: $row"; sleep 45; continue; }
  line=$(printf '%s' "$row" | python3 -c "
import json,sys
r=json.load(sys.stdin)
jobs=', '.join('%s=%s' % (j['name'], (j.get('conclusion') or '-') if j['status']=='completed' else j['status']) for j in r.get('jobs',[]))
print('%s|%s|%s' % (r.get('status','unknown'), r.get('conclusion') or '-', jobs))")
  status="${line%%|*}"; rest="${line#*|}"; conclusion="${rest%%|*}"; jobs="${rest#*|}"
  echo "$(date '+%H:%M:%S')  ${status} ${conclusion}  [${jobs}]"
  case "${status} ${conclusion}" in
    "completed success")
      echo "Build succeeded."
      exit 0
      ;;
    completed*)
      echo "Build did not succeed: ${status} ${conclusion}"
      exit 1
      ;;
  esac
  if [ "$MAX_MINUTES" -gt 0 ] && [ $(( $(date +%s) - START )) -ge $(( MAX_MINUTES * 60 )) ]; then
    echo "Still running after ${MAX_MINUTES} minutes (exit 2 is not a failure)."
    exit 2
  fi
  sleep 45
done
