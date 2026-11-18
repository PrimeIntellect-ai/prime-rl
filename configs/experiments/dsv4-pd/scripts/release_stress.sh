#!/bin/bash
# Cancel the stress-test inference jobs once both level loops have finished.
E=/home/phi/exp/dsv4-pd
BA=63; BP=63
until tail -n +$((BA + 1)) $E/logs/loop_agg.log | grep -qE "all levels done|ended" && tail -n +$((BP + 1)) $E/logs/loop_pd.log | grep -qE "all levels done|ended"; do sleep 60; done
echo "$(date -u +%H:%M) stress loops done; cancelling 344 345"; scancel 344 345
