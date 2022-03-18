#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/" && pwd)"

filelist=$(find "$PROJECT_ROOT" \( -name "*.py" \) -type f)

for f in $filelist; do
  "black" "$f"
  changed=$(git status --porcelain -- "$f" | grep '^ M' | cut -c4-)
  if [[ -n $changed ]]; then
    changelist+=("${changed}")
  fi
done

if [[ ${#changelist[@]} -gt 0 ]]; then
  echo "The following files have clang-format problems"
  for f in "${changelist[@]}"; do
    echo "$f"
  done
  exit 1
fi
