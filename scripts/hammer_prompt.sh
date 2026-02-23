#!/usr/bin/env bash
set -euo pipefail

# Interactive helper that builds a temporary directive from your terminal input and
# then runs `hammer run` (via scripts/hammer.sh) against the chosen repository.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
helper="${repo_root}/scripts/hammer.sh"

target_repo=""
extra_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      if [[ $# -lt 2 ]]; then
        echo "Missing argument for --repo" >&2
        exit 1
      fi
      target_repo="$2"
      shift 2
      ;;
    --*)
      extra_args+=("$1")
      shift
      ;;
    *)
      if [[ -z "$target_repo" ]]; then
        target_repo="$1"
      else
        extra_args+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ -z "$target_repo" ]]; then
  target_repo="$(pwd)"
fi

printf 'Describe the goal for Hammer:\n> '
read -r goal
if [[ -z "$goal" ]]; then
  echo "A goal is required." >&2
  exit 1
fi

constraints=()
while true; do
  printf 'Add a constraint (leave empty to stop):\n> '
  read -r constraint
  [[ -z "$constraint" ]] && break
  constraints+=("$constraint")
done

directive_file="$(mktemp -t hammer-directive-XXXX.md)"
trap 'rm -f "$directive_file"' EXIT
{
  echo "# Directive generated from the terminal"
  echo
  echo "$goal"
  if (( ${#constraints[@]} > 0 )); then
    echo
    echo "## Constraints"
    for c in "${constraints[@]}"; do
      echo "- $c"
    done
  fi
} > "$directive_file"

printf 'Generated directive: %s\n\n' "$directive_file"

cmd=("$helper" run --repo "$target_repo" --directive "$directive_file" "${extra_args[@]}")
printf 'Running: %q\n' "${cmd[@]}"
"${cmd[@]}"
