#!/bin/bash
set -u
rank_log="$RANK_LOG_DIR/rank_${RANK}.log"
exit_file="$RANK_LOG_DIR/exit_rank_${RANK}.txt"
printf 'start %s host %s pid %s\n' "$EPOCHREALTIME" "$(hostname)" "$$" > "$exit_file"

python -m prime_rl.trainer.sft.train @ "$TRAIN_CONFIG_PATH" \
    > >(while IFS= read -r line; do printf '%s %s\n' "$EPOCHREALTIME" "$line"; done >> "$rank_log") 2>&1
rc=$?
printf 'exit %s rc %s\n' "$EPOCHREALTIME" "$rc" >> "$exit_file"
exit "$rc"
