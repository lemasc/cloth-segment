#!/usr/bin/env bash
# Link this clone's datasets/, data/, runs/ to the shared mount.
# Usage: scripts/setup_links.sh [SHARED_ROOT]   (default: /shared/deeplearn/cloth)
set -euo pipefail

shared="${1:-/shared/deeplearn/cloth}"
repo="$(cd "$(dirname "$0")/.." && pwd)"

for name in datasets data runs; do
  target="$shared/$name"
  link="$repo/$name"
  [[ -d "$target" ]] || mkdir -p "$target"
  if [[ -L "$link" ]]; then
    echo "ok      $name -> $(readlink "$link")"
  elif [[ -e "$link" ]]; then
    echo "skip    $name exists and is not a symlink" >&2
  else
    ln -s "$target" "$link"
    echo "linked  $name -> $target"
  fi
done
