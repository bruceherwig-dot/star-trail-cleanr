#!/usr/bin/env bash
# Watch the latest Star Trail CleanR GitHub Actions build until it finishes.
#
# Usage:
#   bash scripts/watch_ci.sh
#
# Polls the latest workflow run every 45 seconds using the gh CLI (authenticated).
# Exits 0 on success, 1 on failure.

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
