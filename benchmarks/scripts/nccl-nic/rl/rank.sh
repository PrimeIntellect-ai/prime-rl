#!/bin/bash
# Usage: rank.sh (run by torchrun --no-python; reads RANK, RANK_LOG_DIR, CONFIG_DIR, TRAIN_HOST)
set -u
rank_log="$RANK_LOG_DIR/rank_${RANK}.log"
exit_file="$RANK_LOG_DIR/exit_rank_${RANK}.txt"
printf 'start %s host %s pid %s\n' "$EPOCHREALTIME" "$(hostname)" "$$" > "$exit_file"

python -m prime_rl.trainer.rl.train @ "$CONFIG_DIR/trainer.json" --rollout_transport.host "$TRAIN_HOST" \
    > >(while IFS= read -r line; do printf '%s %s\n' "$EPOCHREALTIME" "$line"; done >> "$rank_log") 2>&1
rc=$?
printf 'exit %s rc %s\n' "$EPOCHREALTIME" "$rc" >> "$exit_file"
exit "$rc"
