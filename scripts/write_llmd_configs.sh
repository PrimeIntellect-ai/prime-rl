#!/usr/bin/env bash
# Render the llm-d EPP + Envoy configs and the endpoints file for one router
# replica, for no-Kubernetes (file-discovery) operation. The SLURM template
# calls this once per replica, then launches `epp` and `envoy` directly so they
# stay children of the srun shell (so `wait -n` catches a router crash).
#
# Usage:
#   scripts/write_llmd_configs.sh <mode> <router_port> <llmd_dir> [opts] <backend_urls…>
#     mode         : "multi_node" | "disaggregated"
#     router_port  : Envoy listener port (the orchestrator's INFER_URL port)
#     llmd_dir     : output dir for endpoints.yaml, epp.yaml, envoy.yaml
#     --decode-sidecar-port N : (disaggregated) port the pd-sidecar listens on.
#         EPP/Envoy route decode requests here; the sidecar orchestrates remote
#         prefill (x-prefiller-host-port) then forwards decode to vLLM.
#     backend_urls : the per-rank endpoint URLs (already one per DP rank, as the
#         template's external-LB launch enumerates them):
#           multi_node:     http://host:port http://host:port …
#           disaggregated:  --prefill http://h:p … --decode http://h:p …
#
# EPP uses the vllmhttp parser so both the OpenAI path (/v1/chat/completions,
# MITO) and prime-rl's renderer/TITO path (/inference/v1/generate, raw token_ids)
# route and get prefix-cache scoring. Endpoint addresses are resolved to IPv4
# (the file-discovery plugin rejects hostnames). Internal ports (EPP gRPC 9002 /
# health 9003 / metrics 9090, Envoy admin 9901) are loopback-only constants.
set -euo pipefail

mode=$1
router_port=$2
llmd_dir=$3
shift 3
decode_sidecar_port=""
dp_size=1
while [[ $# -gt 0 ]]; do
    case $1 in
        --decode-sidecar-port) decode_sidecar_port=$2; shift 2 ;;
        --dp-size)             dp_size=$2;             shift 2 ;;
        *) break ;;
    esac
done
mkdir -p "$llmd_dir"

# --- endpoints.yaml ---------------------------------------------------------
# One endpoint per DP rank. The template either passes already-per-rank URLs
# (--dp-size 1) or one base URL per node with --dp-size N (consecutive ports
# per rank). For disaggregated decode, the per-rank port is the pd-sidecar's
# (sidecar_base + rank), which proxies to vLLM decode (decode_port + rank).
{
    echo "endpoints:"
    i=0
    role=""
    for tok in "$@"; do
        case "$tok" in
            --prefill) role=prefill ;;
            --decode)  role=decode  ;;
            http://*)
                hp=${tok#http://}
                ip=$(getent hosts "${hp%%:*}" | awk '{print $1; exit}')
                base_port=${hp##*:}
                if [ "$mode" = "disaggregated" ] && [ "$role" = "decode" ] && [ -n "$decode_sidecar_port" ]; then
                    base_port=$decode_sidecar_port
                fi
                for r in $(seq 0 $((dp_size - 1))); do
                    port=$((base_port + r))
                    name="backend-${i}-rank-${r}"
                    [ "$mode" = "disaggregated" ] && name="${role}-${i}-rank-${r}"
                    echo "  - name: ${name}"
                    echo "    address: ${ip}"
                    echo "    port: \"${port}\""
                    echo "    namespace: default"
                    echo "    labels:"
                    echo "      llm-d.ai/pool: prime-rl"
                    [ "$mode" = "disaggregated" ] && echo "      llm-d.ai/role: ${role}"
                done
                i=$((i+1))
                ;;
        esac
    done
} > "$llmd_dir/endpoints.yaml"

# --- epp.yaml ---------------------------------------------------------------
# vllmhttp parser embeds the OpenAI parser and additionally handles
# /inference/v1/generate (token_ids), so MITO + TITO both parse → prefix-cache
# scoring works on both. Parser is selected via requestHandler.parser.pluginRef.
common_header='apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - name: file-disc
    type: file-discovery
    parameters:
      path: '"$llmd_dir"'/endpoints.yaml
      watchFile: true
  - name: vllmhttp-parser
    type: vllmhttp-parser'

epp_footer='requestHandler:
  parser:
    pluginRef: vllmhttp-parser
dataLayer:
  discovery:
    pluginRef: file-disc'

if [ "$mode" = "disaggregated" ]; then
    cat > "$llmd_dir/epp.yaml" <<EOF
${common_header}
  - name: pd-decider
    type: prefix-based-pd-decider
    parameters:
      nonCachedTokens: 16
  - name: profile-handler
    type: disagg-profile-handler
    parameters:
      deciders:
        prefill: pd-decider
  - type: prefill-filter
  - type: decode-filter
  - type: prefix-cache-scorer
  - type: queue-scorer
  - type: kv-cache-utilization-scorer
  - type: active-request-scorer
  - type: max-score-picker
schedulingProfiles:
  - name: prefill
    plugins:
      - pluginRef: prefill-filter
      - pluginRef: prefix-cache-scorer
        weight: 3
      - pluginRef: queue-scorer
        weight: 2
      - pluginRef: kv-cache-utilization-scorer
        weight: 2
      - pluginRef: max-score-picker
  - name: decode
    plugins:
      - pluginRef: decode-filter
      - pluginRef: active-request-scorer
        weight: 2
      - pluginRef: prefix-cache-scorer
        weight: 3
      - pluginRef: max-score-picker
${epp_footer}
EOF
else
    cat > "$llmd_dir/epp.yaml" <<EOF
${common_header}
  - type: queue-scorer
  - type: kv-cache-utilization-scorer
  - type: prefix-cache-scorer
  - type: max-score-picker
  - type: single-profile-handler
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: queue-scorer
        weight: 2
      - pluginRef: kv-cache-utilization-scorer
        weight: 2
      - pluginRef: prefix-cache-scorer
        weight: 3
      - pluginRef: max-score-picker
${epp_footer}
EOF
fi

# --- envoy.yaml -------------------------------------------------------------
# EPP picks the backend (ext_proc) and sets x-gateway-destination-endpoint;
# Envoy's ORIGINAL_DST cluster routes there. Both /v1/ and /inference/v1/ are
# matched so MITO and TITO go through the picker.
cat > "$llmd_dir/envoy.yaml" <<EOF
admin:
  address: { socket_address: { address: 127.0.0.1, port_value: 9901 } }
static_resources:
  listeners:
    - name: prime_rl
      address: { socket_address: { address: 0.0.0.0, port_value: ${router_port} } }
      filter_chains:
        - filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                stat_prefix: prime_rl
                codec_type: AUTO
                stream_idle_timeout: 0s
                request_timeout: 0s
                route_config:
                  virtual_hosts:
                    - name: prime_rl
                      domains: ["*"]
                      routes:
                        - match: { prefix: "/v1/" }
                          route: { cluster: backend_cluster, timeout: 0s, idle_timeout: 86400s }
                        - match: { prefix: "/inference/v1/" }
                          route: { cluster: backend_cluster, timeout: 0s, idle_timeout: 86400s }
                http_filters:
                  - name: envoy.filters.http.ext_proc
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.ext_proc.v3.ExternalProcessor
                      grpc_service:
                        envoy_grpc: { cluster_name: epp_cluster }
                        timeout: 10s
                      processing_mode:
                        request_header_mode: SEND
                        response_header_mode: SEND
                        request_body_mode: FULL_DUPLEX_STREAMED
                        response_body_mode: FULL_DUPLEX_STREAMED
                        request_trailer_mode: SEND
                        response_trailer_mode: SEND
                      message_timeout: 1000s
                  - name: envoy.filters.http.router
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
                      suppress_envoy_headers: true
  clusters:
    - name: backend_cluster
      type: ORIGINAL_DST
      lb_policy: CLUSTER_PROVIDED
      original_dst_lb_config:
        use_http_header: true
        http_header_name: x-gateway-destination-endpoint
      connect_timeout: 5s
    - name: epp_cluster
      type: STRICT_DNS
      typed_extension_protocol_options:
        envoy.extensions.upstreams.http.v3.HttpProtocolOptions:
          "@type": type.googleapis.com/envoy.extensions.upstreams.http.v3.HttpProtocolOptions
          explicit_http_config:
            http2_protocol_options: {}
      load_assignment:
        cluster_name: epp_cluster
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address: { socket_address: { address: 127.0.0.1, port_value: 9002 } }
      connect_timeout: 5s
EOF
