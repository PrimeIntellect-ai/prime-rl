#!/bin/bash
# Usage: comp.sh <name> <log> <command...>
# Runs one component with an epoch-prefixed log and records its pid, start, exit time and rc in $EXIT_DIR/<name>.txt.
set -u
name=$1
log=$2
shift 2
exit_file=$EXIT_DIR/$name.txt
"$@" > >(while IFS= read -r line; do printf '%s %s\n' "$EPOCHREALTIME" "$line"; done >> "$log") 2>&1 &
pid=$!
printf 'start %s host %s pid %s\n' "$EPOCHREALTIME" "$(hostname)" "$pid" > "$exit_file"
wait "$pid"
rc=$?
printf 'exit %s rc %s\n' "$EPOCHREALTIME" "$rc" >> "$exit_file"
exit "$rc"
