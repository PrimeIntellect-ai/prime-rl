#!/usr/bin/env bash
# Render llm-d EPP + Envoy configs and the endpoints file for a single router
# replica. The SLURM template calls this once per replica, then launches `epp`
# and `envoy` directly so they remain children of the srun shell (so `wait -n`
# catches a router crash and tears the job down).
#
# Usage:
#   scripts/write_llmd_configs.sh <mode> <router_port> <llmd_dir> [--decode-sidecar-port N] <backend_args…>
#     mode          : "multi_node" or "disaggregated"
#     router_port   : Envoy listener port (the orchestrator's INFER_URL port)
#     llmd_dir      : directory to write endpoints.yaml, epp.yaml, envoy.yaml
#     --decode-sidecar-port N : (disaggregated only) port the pd-sidecar listens
#       on. EPP/Envoy route decode requests here; the sidecar then orchestrates
#       remote prefill (via x-prefiller-host-port) and forwards decode to vLLM
#       on the original port. Required for canonical llm-d P/D.
#     backend_args… : the same args we'd pass to vllm-router:
#       multi_node:     http://host1:port http://host2:port …
#       disaggregated:  --prefill http://h:p … --decode http://h:p …
#       (For disaggregated, --decode URLs use vLLM's port; the script rewrites
#       them to the sidecar port in endpoints.yaml.)
#
# Endpoint addresses are resolved to IPv4 (the EPP file-discovery plugin
# rejects hostnames). Internal ports (EPP gRPC 9002 / health 9003 /
# metrics 9090, Envoy admin 9901) are hardcoded — they're loopback-only.

set -euo pipefail

mode=$1
router_port=$2
llmd_dir=$3
shift 3
decode_sidecar_port=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --decode-sidecar-port) decode_sidecar_port=$2; shift 2 ;;
        *) break ;;
    esac
done
mkdir -p "$llmd_dir"

#
# endpoints.yaml
#
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
                port=${hp##*:}
                if [ "$mode" = "disaggregated" ]; then
                    name="${role}-${i}"
                    # Decode requests go via pd-sidecar; rewrite to sidecar port.
                    if [ "$role" = "decode" ] && [ -n "$decode_sidecar_port" ]; then
                        port=$decode_sidecar_port
                    fi
                else
                    name="backend-${i}"
                fi
                echo "  - name: ${name}"
                echo "    address: ${ip}"
                echo "    port: \"${port}\""
                echo "    namespace: default"
                echo "    labels:"
                echo "      llm-d.ai/pool: prime-rl"
                if [ "$mode" = "disaggregated" ]; then
                    echo "      llm-d.ai/role: ${role}"
                fi
                i=$((i+1))
                ;;
        esac
    done
} > "$llmd_dir/endpoints.yaml"

#
# epp.yaml — plugin set differs by mode
#   multi_node:    single scheduling profile (optimized-baseline scorer mix)
#   disaggregated: pd-disaggregation profile (separate prefill/decode plugins)
#
common_header='apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - name: file-disc
    type: file-discovery
    parameters:
      path: '"$llmd_dir"'/endpoints.yaml
      watchFile: true'

epp_footer='dataLayer:
  discovery:
    pluginRef: file-disc'

if [ "$mode" = "disaggregated" ]; then
    cat > "$llmd_dir/epp.yaml" <<EOF
${common_header}
  - name: disagg-headers
    type: disagg-headers-handler
  - name: pd-decider
    type: prefix-based-pd-decider
    parameters:
      nonCachedTokens: 8
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

#
# envoy.yaml — identical across modes (EPP picks backend via ext_proc;
# Envoy's ORIGINAL_DST cluster reads x-gateway-destination-endpoint).
#
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
                          typed_per_filter_config:
                            envoy.filters.http.ext_proc:
                              "@type": type.googleapis.com/envoy.config.route.v3.FilterConfig
                              config: {}
                        - match: { prefix: "/inference/v1/" }
                          route: { cluster: backend_cluster, timeout: 0s, idle_timeout: 86400s }
                          typed_per_filter_config:
                            envoy.filters.http.ext_proc:
                              "@type": type.googleapis.com/envoy.config.route.v3.FilterConfig
                              config: {}
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
