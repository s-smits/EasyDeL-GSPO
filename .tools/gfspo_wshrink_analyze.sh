#!/usr/bin/env bash
set -euo pipefail

# Ensure repo is up to date
git fetch --all --prune >/dev/null 2>&1 || true

# Report file (allow override via REPORT env)
if [[ -z "${REPORT:-}" ]]; then
  TS="$(date +%Y%m%d-%H%M%S)"
  REPORT="gfspo-wshrink-commit-analysis-${TS}.txt"
fi

# Header
{
  echo "GFSPO-WShrink Branch Analysis"
  echo "Generated: $(date -Is)"
  echo
  echo "Base branch: $(git rev-parse --abbrev-ref HEAD)"
  echo 'Analyzed range: commits in origin/gfspo-wshrink not in current HEAD'
  echo
} >"${REPORT}"

# Commit range: everything on gfspo-wshrink that is not reachable from current HEAD
# Exclude merge commits and upstream leveling commits by subject.
SKIP_RE=${SKIP_RE:-"merge upstream|upstream/main|upstream main|rebase|integrate best-of-both|adopt upstream|upstream perf|merge remote-tracking|merge pull|merge branch"}
COMMIT_LINES=$(git log --no-merges --reverse --format='%H	%s' HEAD..origin/gfspo-wshrink || true)
FILTERED_SHAS=$(printf "%s\n" "${COMMIT_LINES}" | awk -F '\t' -v re="$SKIP_RE" -v IGNORECASE=1 'NF>=1 { if ($2 !~ re) print $1 }')

COUNT=$(printf "%s\n" "${FILTERED_SHAS}" | sed '/^$/d' | wc -l || true)
echo "Total commits: ${COUNT}" 1>&2
printf "Total commits: %s\n\n" "${COUNT}" >>"${REPORT}"

if [[ "${COUNT}" -eq 0 ]]; then
  echo "No commits to analyze." 1>&2
  echo "No commits to analyze." >>"${REPORT}"
  echo "${REPORT}"
  exit 0
fi

idx=0
printf "%s\n" "${FILTERED_SHAS}" | sed '/^$/d' | while IFS= read -r SHA; do
  idx=$((idx+1))
  SUBJECT=$(git show -s --format=%s "${SHA}")
  AUTHOR=$(git show -s --format=%an "${SHA}")
  AD=$(git show -s --date=iso --format=%ad "${SHA}")
  echo "Processing [${idx}/${COUNT}] ${SHA} - ${SUBJECT}" 1>&2

  {
    echo "=== ${SUBJECT} [${SHA}] ==="
    echo "Author/Date: ${AUTHOR} | ${AD}"
    echo
    echo 'Files/Stats:'
    git show --stat --pretty=format: "${SHA}"
    echo
    echo 'Diff (first 800 lines):'
    git show -p --unified=3 --pretty=format: "${SHA}" | sed -n '1,800p'
    echo
    echo 'Speculative reasoning based on subsequent commits:'
    git log --reverse --format='%h %s' "${SHA}"..origin/gfspo-wshrink | \
      grep -Ei 'fix|revert|bug|stabil|hardening|cleanup|barrier|sync|seed|max_length|mask|cache|import|shape|broadcast|none|alignment|jit|compile|deterministic|debug' | head -n 5 || true
    echo
  } >>"${REPORT}"
done

echo "${REPORT}"


