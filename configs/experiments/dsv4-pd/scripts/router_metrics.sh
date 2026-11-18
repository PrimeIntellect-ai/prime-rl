#!/bin/bash
# Sample the PD router's bidirectional-KV and turn counters once a minute: router_metrics.sh <out.prom>
while true; do
    echo "=== $(date -u +%FT%TZ)"
    curl -s localhost:29000/metrics | grep -E "^vllm_router_pd_(bidir|turn)"
    sleep 60
done >> "$1"
