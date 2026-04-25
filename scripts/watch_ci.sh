#!/usr/bin/env bash
# Watch the latest Star Trail CleanR GitHub Actions build until it finishes.
#
# Usage:
#   bash tools/watch_ci.sh
#
# Polls the latest workflow run for bruceherwig-dot/star-trail-cleanr every
# 45 seconds. Prints a timestamp + status on each check. Exits 0 if the build
# succeeded, 1 if it failed or returned an unexpected status. Designed to be
# run after `git push origin <tag>` as the standard last step before posting
# download links to the user.

REPO="bruceherwig-dot/star-trail-cleanr"
API="https://api.github.com/repos/${REPO}/actions/runs"

run_json=$(curl -sS "${API}?per_page=1") || { echo "API unreachable."; exit 1; }
RUN_ID=$(printf '%s' "$run_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['workflow_runs'][0]['id'])")
TAG=$(printf '%s' "$run_json"    | python3 -c "import json,sys; r=json.load(sys.stdin)['workflow_runs'][0]; print(r.get('head_branch') or r.get('display_title') or 'unknown')")
URL=$(printf '%s' "$run_json"    | python3 -c "import json,sys; print(json.load(sys.stdin)['workflow_runs'][0]['html_url'])")

echo "Watching build for ${TAG} (run ${RUN_ID})"
echo "  ${URL}"
echo

while true; do
  s=$(curl -sS "${API}/${RUN_ID}" | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['status'], r.get('conclusion') or '-')")
  echo "$(date '+%H:%M:%S')  ${s}"
  case "$s" in
    in_progress*|queued*)
      sleep 45
      ;;
    "completed success")
      echo "Build succeeded."
      exit 0
      ;;
    completed*)
      echo "Build did not succeed: ${s}"
      exit 1
      ;;
    *)
      echo "Unexpected status: ${s}"
      exit 1
      ;;
  esac
done
