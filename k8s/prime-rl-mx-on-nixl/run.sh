#!/bin/bash
# Orchestration helper for deploying prime-rl-mx-on-nixl on GB200.
#
# Usage:
#   ./run.sh deploy A   # Scenario A: pure PI, SPG via static host/port
#   ./run.sh deploy B   # Scenario B: MX-mediated SPG rendezvous
#   ./run.sh deploy C   # Scenario B + pipeline replication
#   ./run.sh clean      # Delete all resources
#   ./run.sh logs       # Tail trainer + inference + orchestrator logs
#   ./run.sh status     # Pod status

set -euo pipefail

NS="kavin"
RESOURCES=(
    k8s/prime-rl-mx-on-nixl/config.yaml
    k8s/prime-rl-mx-on-nixl/trainer.yaml
    k8s/prime-rl-mx-on-nixl/inference.yaml
    k8s/prime-rl-mx-on-nixl/orchestrator.yaml
)

MX_ENV='- name: PRIME_RL_MX_RENDEZVOUS
              value: "modelexpress-server.kavin.svc.cluster.local:8001"
            - name: PRIME_RL_MX_MODEL_NAME
              value: "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"'

MX_PIPE_ENV='- name: PRIME_RL_MX_PIPELINE_REPLICATION
              value: "1"'

cmd="${1:-}"
scenario="${2:-A}"

case "$cmd" in
    deploy)
        echo "=== deploying scenario ${scenario} ==="
        # Apply base manifests
        for f in "${RESOURCES[@]}"; do
            kubectl apply -f "$f"
        done

        if [[ "$scenario" == "B" || "$scenario" == "C" ]]; then
            echo "=== enabling MX rendezvous on trainer + inference ==="
            kubectl -n "$NS" set env statefulset/prime-rl-mx-on-nixl-trainer \
                PRIME_RL_MX_RENDEZVOUS="modelexpress-server.kavin.svc.cluster.local:8001" \
                PRIME_RL_MX_MODEL_NAME="PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
            kubectl -n "$NS" set env statefulset/prime-rl-mx-on-nixl-inference \
                PRIME_RL_MX_RENDEZVOUS="modelexpress-server.kavin.svc.cluster.local:8001" \
                PRIME_RL_MX_MODEL_NAME="PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
        fi

        if [[ "$scenario" == "C" ]]; then
            echo "=== enabling pipeline replication ==="
            kubectl -n "$NS" set env statefulset/prime-rl-mx-on-nixl-inference \
                PRIME_RL_MX_PIPELINE_REPLICATION="1"
        fi

        echo "=== waiting for pods ==="
        kubectl -n "$NS" rollout status --timeout=600s statefulset/prime-rl-mx-on-nixl-trainer
        kubectl -n "$NS" rollout status --timeout=600s statefulset/prime-rl-mx-on-nixl-inference
        kubectl -n "$NS" rollout status --timeout=600s deployment/prime-rl-mx-on-nixl-orchestrator

        echo "=== deployed. use 'run.sh logs' to watch output ==="
        ;;
    clean)
        echo "=== deleting resources ==="
        kubectl -n "$NS" delete statefulset,deployment,svc,configmap \
            -l 'app in (prime-rl-mx-on-nixl-trainer,prime-rl-mx-on-nixl-inference,prime-rl-mx-on-nixl-orchestrator)' \
            --ignore-not-found
        kubectl -n "$NS" delete configmap prime-rl-mx-on-nixl-config --ignore-not-found
        ;;
    status)
        kubectl -n "$NS" get pods -l 'app in (prime-rl-mx-on-nixl-trainer,prime-rl-mx-on-nixl-inference,prime-rl-mx-on-nixl-orchestrator)' -o wide
        ;;
    logs)
        echo "=== recent trainer logs ==="
        kubectl -n "$NS" logs prime-rl-mx-on-nixl-trainer-0 --tail=50 2>&1 || true
        echo ""
        echo "=== recent inference logs ==="
        kubectl -n "$NS" logs prime-rl-mx-on-nixl-inference-0 --tail=50 2>&1 || true
        echo ""
        echo "=== recent orchestrator logs ==="
        kubectl -n "$NS" logs -l app=prime-rl-mx-on-nixl-orchestrator --tail=50 2>&1 || true
        ;;
    *)
        echo "usage: $0 {deploy A|B|C | clean | status | logs}"
        exit 2
        ;;
esac
