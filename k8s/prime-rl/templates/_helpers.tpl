{{/*
Expand the name of the chart.
*/}}
{{- define "prime-rl.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Resolve the immutable image reference generated for DGD, with the native chart
repository/tag remaining as the fallback for statefulset mode.
*/}}
{{- define "prime-rl.image" -}}
{{- if .Values.image.reference -}}
{{- .Values.image.reference -}}
{{- else -}}
{{- printf "%s:%s" .Values.image.repository .Values.image.tag -}}
{{- end -}}
{{- end }}

{{/*
Reuse a supplied shared claim or derive the chart-managed claim name.
*/}}
{{- define "prime-rl.storageClaimName" -}}
{{- default (printf "%s-shared-data" .Release.Name) .Values.storage.existingClaim -}}
{{- end }}

{{- define "prime-rl.inferenceUrls" -}}
{{- if eq .Values.inference.mode "dynamoGraph" -}}
{{- printf "http://%s-frontend.%s.svc.cluster.local:8000/v1" .Release.Name .Values.namespace -}}
{{- else -}}
{{- $releaseName := .Release.Name -}}
{{- $namespace := .Values.namespace -}}
{{- $port := int .Values.inference.service.port -}}
{{- $replicas := int .Values.inference.replicas -}}
{{- $urls := list -}}
{{- range $i := until $replicas -}}
{{- $url := printf "http://%s-inference-%d.%s-inference-headless.%s.svc.cluster.local:%d/v1" $releaseName $i $releaseName $namespace $port -}}
{{- $urls = append $urls $url -}}
{{- end -}}
{{- $urls | join "," -}}
{{- end -}}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "prime-rl.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "prime-rl.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "prime-rl.labels" -}}
helm.sh/chart: {{ include "prime-rl.chart" . }}
{{ include "prime-rl.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "prime-rl.selectorLabels" -}}
app.kubernetes.io/name: {{ include "prime-rl.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Component labels
*/}}
{{- define "prime-rl.componentLabels" -}}
app: prime-rl
{{- if .Values.config.example }}
example: {{ .Values.config.example }}
{{- end }}
{{- end }}
