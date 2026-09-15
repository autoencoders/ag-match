#!/usr/bin/env sh
# Refresh the `package` branch: the contents of src/ag_match at the root, nothing else.
# Apps vendor it with:
#   git subtree add --prefix=<path> https://github.com/autoencoders/ag-match.git package --squash
# Run from any checkout after the source changes land, e.g. ./scripts/publish-package-branch.sh main
set -eu
src="${1:-HEAD}"
sha="$(git subtree split --prefix=src/ag_match "$src")"
git push --force-with-lease origin "$sha:refs/heads/package"
echo "package branch now at $(git rev-parse --short "$sha")"
