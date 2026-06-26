{{/* Release-scoped name fragments and label sets. */}}

{{- define "prime-rl-llmd.fullname" -}}
{{- .Release.Name -}}
{{- end -}}

{{- define "prime-rl-llmd.labels" -}}
app.kubernetes.io/name: prime-rl-llmd
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "prime-rl-llmd.selectorLabels" -}}
app.kubernetes.io/name: prime-rl-llmd
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/* mooncake master DNS name. Inference pods use this for the master + metadata server. */}}
{{- define "prime-rl-llmd.mooncakeMasterHost" -}}
{{ .Release.Name }}-mooncake.{{ .Values.namespace }}.svc.cluster.local
{{- end -}}

{{/* Common pod spec env: secrets + telemetry. */}}
{{- define "prime-rl-llmd.commonEnv" -}}
- name: POD_NAME
  valueFrom:
    fieldRef:
      fieldPath: metadata.name
- name: POD_IP
  valueFrom:
    fieldRef:
      fieldPath: status.podIP
- name: POD_NAMESPACE
  valueFrom:
    fieldRef:
      fieldPath: metadata.namespace
{{- if .Values.secrets.enabled }}
- name: HF_TOKEN
  valueFrom:
    secretKeyRef:
      name: {{ .Values.secrets.name }}
      key: hf-token
      optional: true
- name: WANDB_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Values.secrets.name }}
      key: wandb-api-key
      optional: true
{{- end }}
{{- end -}}
