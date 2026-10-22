#!/bin/bash
# Run each line of the given file as launch.sh arguments, serially.
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  if [ "$(date -u +%H%M)" \> "0219" ] && [ "$(date -u +%H%M)" \< "1200" ]; then echo "past 02:20, stop"; break; fi
  echo "$(date -u +%T) start: $line"
  eval ./launch.sh $line
  echo "$(date -u +%T) done: $(tail -1 ~/tmp/dist-moe-sft/$(echo $line | cut -d' ' -f1).out)"
done < "$1"
