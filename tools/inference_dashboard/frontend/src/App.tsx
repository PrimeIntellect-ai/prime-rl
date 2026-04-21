import { useEffect, useMemo, useState } from "react";
import { ChevronLeftIcon, ChevronRightIcon } from "@radix-ui/react-icons";
import {
  Badge,
  Button,
  Card,
  Flex,
  Heading,
  Text,
  TextField,
} from "@radix-ui/themes";
import {
  Bar,
  BarChart,
  Area,
  AreaChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { PodSnapshot, Snapshot, TimeSeriesPoint } from "./types";

type FormState = {
  job_id: number;
  num_cold_prefill_nodes_per_pod: number;
  num_prefill_nodes_per_pod: number;
  num_decode_nodes_per_pod: number;
  num_replicas: number;
};

const initialForm: FormState = {
  job_id: 3781,
  num_cold_prefill_nodes_per_pod: 0,
  num_prefill_nodes_per_pod: 4,
  num_decode_nodes_per_pod: 4,
  num_replicas: 3,
};

const NODE_SERIES_COLORS = ["#f5f5f5", "#d7d7d7", "#bcbcbc", "#a0a0a0", "#888888", "#727272"];

export function App() {
  const [form, setForm] = useState<FormState>(initialForm);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedPodIndex, setSelectedPodIndex] = useState(0);

  useEffect(() => {
    if (snapshot === null) {
      return;
    }

    const eventSource = new EventSource(`/api/jobs/${snapshot.job_id}/stream`);
    eventSource.onmessage = (event) => setSnapshot(JSON.parse(event.data));
    eventSource.onerror = () => setError("Realtime stream disconnected. The latest snapshot is still shown.");
    return () => eventSource.close();
  }, [snapshot?.job_id]);

  useEffect(() => {
    if (snapshot === null) {
      return;
    }
    setSelectedPodIndex((current) => Math.min(current, Math.max(snapshot.pods.length - 1, 0)));
  }, [snapshot]);

  const kvDistribution = useMemo(() => {
    if (snapshot === null) {
      return [];
    }
    return snapshot.pods.map((pod) => ({
      name: `Pod ${pod.pod_index}`,
      value: Number((pod.max_kv_cache_usage * 100).toFixed(2)),
    }));
  }, [snapshot]);

  const selectedPod = snapshot?.pods[selectedPodIndex] ?? null;

  async function connect() {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/topology/from-slurm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form,
          num_prefill_replicas_per_pod: 1,
          num_decode_replicas_per_pod: 1,
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const snapshotResponse = await fetch(`/api/jobs/${form.job_id}/snapshot`);
      if (!snapshotResponse.ok) {
        throw new Error(await snapshotResponse.text());
      }
      setSnapshot(await snapshotResponse.json());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to connect");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Prime RL Side Utility</p>
          <h1>Inference Observatory</h1>
          <p className="lede">
            Live cluster visibility for router health, prefill throughput, decode throughput, KV cache pressure,
            request completion, and NIXL KV transfer behavior.
          </p>
          <div className="hero-badges">
            <span>Reactive stream</span>
            <span>Slurm discovery</span>
            <span>Pod-first views</span>
          </div>
        </div>

        <Card className="connect-card">
          <Flex justify="between" align="start" gap="3">
            <div>
              <p className="eyebrow">Connect</p>
              <Heading size="5">from_slurm</Heading>
            </div>
            <Badge color={loading ? "amber" : "cyan"} variant="soft">
              {loading ? "Loading" : "Ready"}
            </Badge>
          </Flex>

          <div className="form-grid">
            <Field
              label="Job ID"
              value={form.job_id}
              onChange={(value) => setForm((current) => ({ ...current, job_id: Number(value) }))}
            />
            <Field
              label="Cold / pod"
              value={form.num_cold_prefill_nodes_per_pod}
              onChange={(value) =>
                setForm((current) => ({ ...current, num_cold_prefill_nodes_per_pod: Number(value) }))
              }
            />
            <Field
              label="Prefill / pod"
              value={form.num_prefill_nodes_per_pod}
              onChange={(value) => setForm((current) => ({ ...current, num_prefill_nodes_per_pod: Number(value) }))}
            />
            <Field
              label="Decode / pod"
              value={form.num_decode_nodes_per_pod}
              onChange={(value) => setForm((current) => ({ ...current, num_decode_nodes_per_pod: Number(value) }))}
            />
            <Field
              label="Pods"
              value={form.num_replicas}
              onChange={(value) => setForm((current) => ({ ...current, num_replicas: Number(value) }))}
            />
          </div>

          <Button className="primary-button" size="3" type="button" onClick={connect} disabled={loading}>
            {loading ? "Connecting..." : "Connect to Job"}
          </Button>
          {error ? <p className="error-text">{error}</p> : null}
        </Card>
      </header>

      {snapshot === null ? (
        <section className="empty-panel">
          <h2>No job connected yet</h2>
          <p>The UI will start streaming metrics as soon as a Slurm topology resolves and the first snapshot lands.</p>
        </section>
      ) : (
        <main className="dashboard">
          <section className="summary-grid">
            <SummaryCard label="Routers Healthy" value={`${snapshot.summary.healthy_routers}/${snapshot.summary.total_routers}`} />
            <SummaryCard label="Prefill tok/s" value={rate(snapshot.summary.total_prefill_tokens_per_second)} />
            <SummaryCard label="Decode tok/s" value={rate(snapshot.summary.total_decode_tokens_per_second)} />
            <SummaryCard label="Requests / s" value={rate(snapshot.summary.total_requests_per_second)} />
            <SummaryCard label="Running" value={integer(snapshot.summary.total_running_requests)} />
            <SummaryCard label="Waiting" value={integer(snapshot.summary.total_waiting_requests)} />
            <SummaryCard label="KV Max" value={percent(snapshot.summary.max_kv_cache_usage)} />
            <SummaryCard
              label="Healthy Nodes"
              value={`${
                snapshot.summary.healthy_cold_prefill_nodes +
                snapshot.summary.healthy_prefill_nodes +
                snapshot.summary.healthy_decode_nodes
              }/${
                snapshot.summary.total_cold_prefill_nodes +
                snapshot.summary.total_prefill_nodes +
                snapshot.summary.total_decode_nodes
              }`}
            />
          </section>

          <section className="chart-grid">
            <ChartCard title="Cluster Throughput">
              <div className="chart-frame">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={mergeSeries(
                    snapshot.history,
                    ["prefill_tokens_per_second", "decode_tokens_per_second", "requests_per_second"],
                    {
                      prefill_tokens_per_second: snapshot.summary.total_prefill_tokens_per_second,
                      decode_tokens_per_second: snapshot.summary.total_decode_tokens_per_second,
                      requests_per_second: snapshot.summary.total_requests_per_second,
                    },
                  )}
                >
                  <defs>
                    <linearGradient id="cluster-prefill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#34d1bf" stopOpacity={0.7} />
                      <stop offset="95%" stopColor="#34d1bf" stopOpacity={0.04} />
                    </linearGradient>
                    <linearGradient id="cluster-decode" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#5ca9ff" stopOpacity={0.7} />
                      <stop offset="95%" stopColor="#5ca9ff" stopOpacity={0.04} />
                    </linearGradient>
                    <linearGradient id="cluster-requests" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f6b84b" stopOpacity={0.7} />
                      <stop offset="95%" stopColor="#f6b84b" stopOpacity={0.04} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="rgba(148,178,221,0.12)" />
                  <XAxis dataKey="time" tick={{ fill: "#8d949e" }} tickLine={false} axisLine={false} />
                  <YAxis
                    width={72}
                    tickFormatter={formatAxisNumber}
                    tick={{ fill: "#8d949e" }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip formatter={formatTooltipValue} contentStyle={{ background: "#0f1115", border: "1px solid rgba(255,255,255,0.08)" }} />
                  <Area type="monotone" dataKey="prefill_tokens_per_second" stroke="#f5f5f5" fill="url(#cluster-prefill)" strokeWidth={2} />
                  <Area type="monotone" dataKey="decode_tokens_per_second" stroke="#bdbdbd" fill="url(#cluster-decode)" strokeWidth={2} />
                  <Area type="monotone" dataKey="requests_per_second" stroke="#7e7e7e" fill="url(#cluster-requests)" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
              </div>
            </ChartCard>

            <ChartCard title="Pod KV Envelope">
              <div className="chart-frame">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={kvDistribution} dataKey="value" nameKey="name" outerRadius={90} innerRadius={45} paddingAngle={3}>
                    {kvDistribution.map((entry, index) => (
                      <Cell key={entry.name} fill={["#34d1bf", "#5ca9ff", "#f6b84b", "#ff7c8a"][index % 4]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={formatPercentTooltipValue} contentStyle={{ background: "#0f1115", border: "1px solid rgba(255,255,255,0.08)" }} />
                </PieChart>
              </ResponsiveContainer>
              </div>
            </ChartCard>
          </section>

          <section className="section-card">
            <div className="section-header">
              <div>
                <p className="eyebrow">Pods</p>
                <Heading size="6">Pod Dashboard</Heading>
              </div>
              <Flex gap="2" align="center">
                <Button
                  variant="soft"
                  onClick={() => setSelectedPodIndex((current) => Math.max(current - 1, 0))}
                  disabled={selectedPodIndex === 0}
                >
                  <ChevronLeftIcon />
                </Button>
                <Text size="2" color="gray">
                  Pod {selectedPodIndex + 1} / {snapshot.pods.length}
                </Text>
                <Button
                  variant="soft"
                  onClick={() => setSelectedPodIndex((current) => Math.min(current + 1, snapshot.pods.length - 1))}
                  disabled={selectedPodIndex === snapshot.pods.length - 1}
                >
                  <ChevronRightIcon />
                </Button>
              </Flex>
            </div>
            {selectedPod ? (
              <PodCarouselCard pod={selectedPod} history={snapshot.pod_history?.[selectedPod.id] ?? {}} />
            ) : null}
          </section>
        </main>
      )}
    </div>
  );
}

function SummaryCard(props: { label: string; value: string }) {
  return (
    <Card className="summary-card">
      <Text size="2" color="gray">
        {props.label}
      </Text>
      <Heading size="7">{props.value}</Heading>
    </Card>
  );
}

function ChartCard(props: { title: string; children: React.ReactNode }) {
  return (
    <Card className="section-card chart-card">
      <div className="chart-header">
        <Heading size="3">{props.title}</Heading>
      </div>
      {props.children}
    </Card>
  );
}

function PodCarouselCard({
  pod,
  history,
}: {
  pod: PodSnapshot;
  history: Record<string, TimeSeriesPoint[]>;
}) {
  const prefillRunningSeries = mergeSeries(history, ["prefill_running_requests"], {
    prefill_running_requests: pod.prefill_replica.total_running_requests,
  });
  const coldPrefillRunningSeries = mergeSeries(history, ["cold_prefill_running_requests"], {
    cold_prefill_running_requests: pod.cold_prefill_running_requests,
  });
  const decodeRunningSeries = mergeSeries(history, ["decode_running_requests"], {
    decode_running_requests: pod.decode_replica.total_running_requests,
  });
  const requestsSeries = mergeSeries(history, ["requests_per_second"], {
    requests_per_second: pod.total_requests_per_second,
  });
  const prefillThroughputSeries = mergeSeries(history, ["prefill_tokens_per_second"], {
    prefill_tokens_per_second: pod.total_prefill_tokens_per_second,
  });
  const decodeThroughputSeries = mergeSeries(history, ["decode_tokens_per_second"], {
    decode_tokens_per_second: pod.total_decode_tokens_per_second,
  });
  const prefixCacheHitRateSeries = mergeSeries(history, ["prefix_cache_hit_rate"], {
    prefix_cache_hit_rate: pod.prefill_prefix_cache_hit_rate,
  });
  const prefixCacheHitRateWithCpuSeries = mergeSeries(
    history,
    ["prefix_cache_hit_rate", "cpu_prefix_cache_hit_rate"],
    {
      prefix_cache_hit_rate: pod.prefill_prefix_cache_hit_rate,
      cpu_prefix_cache_hit_rate: pod.prefill_cpu_prefix_cache_hit_rate,
    },
  );
  const hasCpuPrefixCacheHitRate =
    (history.cpu_prefix_cache_hit_rate?.length ?? 0) > 0 || pod.prefill_cpu_prefix_cache_hit_rate !== null;
  const prefixCacheChartSeries = hasCpuPrefixCacheHitRate ? prefixCacheHitRateWithCpuSeries : prefixCacheHitRateSeries;
  const prefixCacheChartAreas: Array<[string, string, string]> = hasCpuPrefixCacheHitRate
    ? [
        ["prefix_cache_hit_rate", "#c7c7c7", "url(#pod-prefix-cache)"],
        ["cpu_prefix_cache_hit_rate", "#8f8f8f", "url(#pod-prefix-cache-cpu)"],
      ]
    : [["prefix_cache_hit_rate", "#c7c7c7", "url(#pod-prefix-cache)"]];
  const prefixCacheChartGradients: Array<[string, string]> = hasCpuPrefixCacheHitRate
    ? [
        ["pod-prefix-cache", "#c7c7c7"],
        ["pod-prefix-cache-cpu", "#8f8f8f"],
      ]
    : [["pod-prefix-cache", "#c7c7c7"]];
  const prefillKvSeries = mergeNodeKvSeries(pod.prefill_nodes, history, "P");
  const decodeKvSeries = mergeNodeKvSeries(pod.decode_nodes, history, "D");
  const prefillWaitingSeries = mergeSeries(history, ["prefill_waiting_requests"], {
    prefill_waiting_requests: pod.prefill_waiting_requests,
  });
  const coldPrefillWaitingSeries = mergeSeries(history, ["cold_prefill_waiting_requests"], {
    cold_prefill_waiting_requests: pod.cold_prefill_waiting_requests,
  });
  const decodeWaitingSeries = mergeSeries(history, ["decode_waiting_requests"], {
    decode_waiting_requests: pod.decode_waiting_requests,
  });
  const nixlSeries = mergeSeries(history, ["nixl_transfer_time_seconds"], {
    nixl_transfer_time_seconds: pod.decode_replica.avg_nixl_transfer_time_seconds ?? 0,
  });

  return (
    <Card className="pod-card">
      <header className="pod-header">
        <div>
          <p className="eyebrow">Pod {pod.pod_index}</p>
          <Heading size="4">{pod.router_url}</Heading>
        </div>
        <Badge color={pod.router_healthy ? "mint" : "red"} variant="soft" size="2">
          {pod.router_healthy ? "Router healthy" : "Router down"}
        </Badge>
      </header>

      <div className="pod-metrics">
        <Metric label="Prefill tok/s" value={rate(pod.total_prefill_tokens_per_second)} />
        <Metric label="Decode tok/s" value={rate(pod.total_decode_tokens_per_second)} />
        <Metric label="Req/s" value={rate(pod.total_requests_per_second)} />
        <Metric label="KV max" value={percent(pod.max_kv_cache_usage)} />
        <Metric
          label="NIXL ms"
          value={
            pod.decode_replica.avg_nixl_transfer_time_seconds === null
              ? "—"
              : formatDisplayNumber(pod.decode_replica.avg_nixl_transfer_time_seconds * 1000)
          }
        />
        <Metric label="Alerts" value={integer(pod.alerts.length)} />
      </div>

      <div className="pod-focus-grid">
        <PodTimeChart
          title="Prefill Running"
          data={prefillRunningSeries}
          gradients={[["pod-running-prefill", "#f4f4f4"]]}
          areas={[["prefill_running_requests", "#f4f4f4", "url(#pod-running-prefill)"]]}
        />
        <PodTimeChart
          title="Prefill Waiting"
          data={prefillWaitingSeries}
          gradients={[["pod-waiting-prefill", "#d6d6d6"]]}
          areas={[["prefill_waiting_requests", "#d6d6d6", "url(#pod-waiting-prefill)"]]}
        />
        <PodTimeChart
          title="Decode Running"
          data={decodeRunningSeries}
          gradients={[["pod-running-decode", "#a8a8a8"]]}
          areas={[["decode_running_requests", "#a8a8a8", "url(#pod-running-decode)"]]}
        />
        <PodTimeChart
          title="Decode Waiting"
          data={decodeWaitingSeries}
          gradients={[["pod-waiting-decode", "#8f8f8f"]]}
          areas={[["decode_waiting_requests", "#8f8f8f", "url(#pod-waiting-decode)"]]}
        />
        <PodTimeChart
          title="Prefill Throughput"
          data={prefillThroughputSeries}
          gradients={[["pod-prefill", "#f4f4f4"]]}
          areas={[["prefill_tokens_per_second", "#f4f4f4", "url(#pod-prefill)"]]}
        />
        <PodTimeChart
          title="Decode Throughput"
          data={decodeThroughputSeries}
          gradients={[["pod-decode", "#a8a8a8"]]}
          areas={[["decode_tokens_per_second", "#a8a8a8", "url(#pod-decode)"]]}
        />
        <PodTimeChart
          title="Prefill Prefix Cache Hit Rate"
          data={prefixCacheChartSeries}
          gradients={prefixCacheChartGradients}
          areas={prefixCacheChartAreas}
          fullWidth
          valueFormat="percent"
        />
        <PodNodeChart
          title="Prefill KV Cache by Node"
          data={prefillKvSeries.data}
          lines={prefillKvSeries.lines}
        />
        <PodNodeChart
          title="Decode KV Cache by Node"
          data={decodeKvSeries.data}
          lines={decodeKvSeries.lines}
        />
        {pod.cold_prefill_nodes.length > 0 ? (
          <>
            <PodTimeChart
              title="Cold Prefill Waiting"
              data={coldPrefillWaitingSeries}
              gradients={[["pod-cold-waiting", "#d6d6d6"]]}
              areas={[["cold_prefill_waiting_requests", "#d6d6d6", "url(#pod-cold-waiting)"]]}
            />
            <PodTimeChart
              title="Cold Prefill Running"
              data={coldPrefillRunningSeries}
              gradients={[["pod-cold-running", "#f4f4f4"]]}
              areas={[["cold_prefill_running_requests", "#f4f4f4", "url(#pod-cold-running)"]]}
            />
          </>
        ) : null}
        <PodTimeChart
          title="Completed Requests"
          data={requestsSeries}
          gradients={[["pod-requests", "#cfcfcf"]]}
          areas={[["requests_per_second", "#cfcfcf", "url(#pod-requests)"]]}
          chart="bar"
        />
        <PodTimeChart
          title="KV Cache Transfer Time"
          data={nixlSeries}
          gradients={[["pod-nixl", "#b3b3b3"]]}
          areas={[["nixl_transfer_time_ms", "#b3b3b3", "url(#pod-nixl)"]]}
        />
      </div>

      {pod.alerts.length > 0 ? (
        <div className="alert-stack">
          {pod.alerts.map((alert, index) => (
            <div key={`${alert.target}-${index}`} className={`alert-card ${alert.severity}`}>
              {alert.message}
            </div>
          ))}
        </div>
      ) : null}
    </Card>
  );
}

function PodTimeChart({
  title,
  data,
  gradients,
  areas,
  chart,
  fullWidth,
  valueFormat,
}: {
  title: string;
  data: Array<Record<string, number | string>>;
  gradients: Array<[string, string]>;
  areas: Array<[string, string, string]>;
  chart?: "area" | "bar";
  fullWidth?: boolean;
  valueFormat?: "number" | "percent";
}) {
  const cardClassName = fullWidth ? "section-card chart-card chart-card-full" : "section-card chart-card";
  const axisFormatter = valueFormat === "percent" ? formatPercentAxisNumber : formatAxisNumber;
  const tooltipFormatter = valueFormat === "percent" ? formatPercentTooltipValue : formatTooltipValue;

  if (data.length === 0) {
    return (
      <Card className={cardClassName}>
        <div className="chart-header">
          <Heading size="3">{title}</Heading>
        </div>
        <div className="chart-empty">Waiting for enough live samples...</div>
      </Card>
    );
  }

  return (
    <Card className={cardClassName}>
      <div className="chart-header">
        <Heading size="3">{title}</Heading>
      </div>
      <div className="chart-frame">
        <ResponsiveContainer width="100%" height="100%">
          {chart === "bar" ? (
            <BarChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 4 }}>
              <defs>
                {gradients.map(([id, color]) => (
                  <linearGradient key={id} id={id} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={color} stopOpacity={0.9} />
                    <stop offset="95%" stopColor={color} stopOpacity={0.28} />
                  </linearGradient>
                ))}
              </defs>
              <CartesianGrid stroke="rgba(255,255,255,0.07)" vertical={false} />
              <XAxis dataKey="time" tick={{ fill: "#8d949e" }} tickLine={false} axisLine={false} />
              <YAxis
                width={72}
                tickFormatter={axisFormatter}
                tick={{ fill: "#8d949e" }}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip formatter={tooltipFormatter} contentStyle={{ background: "#0f1115", border: "1px solid rgba(255,255,255,0.08)" }} />
              {areas.map(([key, _stroke, fill]) => (
                <Bar key={key} dataKey={key} fill={fill} radius={[8, 8, 0, 0]} />
              ))}
            </BarChart>
          ) : (
            <AreaChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 4 }}>
              <defs>
                {gradients.map(([id, color]) => (
                  <linearGradient key={id} id={id} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={color} stopOpacity={0.7} />
                    <stop offset="95%" stopColor={color} stopOpacity={0.04} />
                  </linearGradient>
                ))}
              </defs>
              <CartesianGrid stroke="rgba(255,255,255,0.07)" vertical={false} />
              <XAxis dataKey="time" tick={{ fill: "#8d949e" }} tickLine={false} axisLine={false} />
              <YAxis
                width={72}
                tickFormatter={axisFormatter}
                tick={{ fill: "#8d949e" }}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip formatter={tooltipFormatter} contentStyle={{ background: "#0f1115", border: "1px solid rgba(255,255,255,0.08)" }} />
              {areas.map(([key, stroke, fill]) => (
                <Area key={key} type="monotone" dataKey={key} stroke={stroke} fill={fill} strokeWidth={2} />
              ))}
            </AreaChart>
          )}
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

function PodNodeChart({
  title,
  data,
  lines,
}: {
  title: string;
  data: Array<Record<string, number | string>>;
  lines: Array<{ key: string; label: string; color: string; strokeDasharray?: string }>;
}) {
  if (data.length === 0 || lines.length === 0) {
    return (
      <ChartCard title={title}>
        <div className="chart-empty">Waiting for enough live samples...</div>
      </ChartCard>
    );
  }

  return (
    <ChartCard title={title}>
      <div className="chart-frame">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 4 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.07)" vertical={false} />
            <XAxis dataKey="time" tick={{ fill: "#8d949e" }} tickLine={false} axisLine={false} />
            <YAxis
              width={72}
              tickFormatter={formatAxisNumber}
              tick={{ fill: "#8d949e" }}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip formatter={formatPercentTooltipValue} contentStyle={{ background: "#0f1115", border: "1px solid rgba(255,255,255,0.08)" }} />
            {lines.map((line) => (
              <Line
                key={line.key}
                type="monotone"
                dataKey={line.key}
                name={line.label}
                stroke={line.color}
                strokeDasharray={line.strokeDasharray}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="chart-legend">
        {lines.map((line) => (
          <span key={line.key} className="chart-legend-item">
            <span className="chart-legend-swatch" style={{ background: line.color }} />
            {line.label}
          </span>
        ))}
      </div>
    </ChartCard>
  );
}

function Metric(props: { label: string; value: string }) {
  return (
    <div className="metric-tile">
      <Text size="2" color="gray">
        {props.label}
      </Text>
      <Heading size="5">{props.value}</Heading>
    </div>
  );
}

function Field(props: { label: string; value: number; onChange: (value: string) => void }) {
  return (
    <label className="field">
      <Text size="2" color="gray">
        {props.label}
      </Text>
      <TextField.Root size="3" type="number" min={1} value={props.value} onChange={(event) => props.onChange(event.target.value)} />
    </label>
  );
}

function mergeNodeKvSeries(
  nodes: PodSnapshot["prefill_nodes"],
  seriesByKey: Record<string, TimeSeriesPoint[]>,
  prefix: string,
) {
  const lineSpecs = nodes.flatMap((node, index): Array<{
    historyKey: string;
    dataKey: string;
    label: string;
    color: string;
    fallbackValue: number | null;
    strokeDasharray?: string;
  }> => {
    const color = NODE_SERIES_COLORS[index % NODE_SERIES_COLORS.length];
    const specs: Array<{
      historyKey: string;
      dataKey: string;
      label: string;
      color: string;
      fallbackValue: number | null;
      strokeDasharray?: string;
    }> = [
      {
        historyKey: `node_kv_cache_usage:${node.id}`,
        dataKey: `${prefix.toLowerCase()}_${index}`,
        label: `${prefix}${index}`,
        color,
        fallbackValue: node.kv_cache_usage_max,
        strokeDasharray: undefined as string | undefined,
      },
    ];
    const hasCpuHistory = (seriesByKey[`node_cpu_kv_cache_usage:${node.id}`]?.length ?? 0) > 0;
    if (hasCpuHistory || node.cpu_kv_cache_usage_max !== null) {
      specs.push({
        historyKey: `node_cpu_kv_cache_usage:${node.id}`,
        dataKey: `${prefix.toLowerCase()}_${index}_cpu`,
        label: `${prefix}${index} CPU`,
        color,
        fallbackValue: node.cpu_kv_cache_usage_max,
        strokeDasharray: "6 4",
      });
    }
    return specs;
  });

  const visibleLineSpecs = lineSpecs.filter(
    (line) => (seriesByKey[line.historyKey]?.length ?? 0) > 0 || line.fallbackValue !== null,
  );

  const timestamps = Array.from(
    new Set(visibleLineSpecs.flatMap((line) => (seriesByKey[line.historyKey] ?? []).map((point) => point.timestamp))),
  ).sort((left, right) => left - right);

  if (timestamps.length === 0 && visibleLineSpecs.every((line) => line.fallbackValue === null)) {
    return { data: [], lines: [] };
  }

  if (timestamps.length === 0) {
    timestamps.push(Date.now() / 1000);
  }

  if (timestamps.length === 1) {
    timestamps.unshift(timestamps[0] - 2);
  }

  const data = timestamps.map((timestamp) => {
    const row: Record<string, number | string> = {
      time: new Date(timestamp * 1000).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }),
    };
    for (const line of visibleLineSpecs) {
      row[line.dataKey] =
        getValueForTimestamp(seriesByKey[line.historyKey] ?? [], timestamp, line.fallbackValue ?? 0) * 100;
    }
    return row;
  });

  const firstLiveIndex = data.findIndex((row) =>
    visibleLineSpecs.some((line) => Math.abs(Number(row[line.dataKey] ?? 0)) > 1e-9),
  );

  return {
    data: firstLiveIndex <= 0 ? data : data.slice(firstLiveIndex - 1),
    lines: visibleLineSpecs.map((line) => ({
      key: line.dataKey,
      label: line.label,
      color: line.color,
      strokeDasharray: line.strokeDasharray,
    })),
  };
}

function mergeSeries(
  seriesByKey: Record<string, TimeSeriesPoint[]>,
  keys: string[],
  fallbackValues: Record<string, number | null>,
) {
  const timestamps = Array.from(
    new Set(keys.flatMap((key) => (seriesByKey[key] ?? []).map((point) => point.timestamp))),
  ).sort((left, right) => left - right);

  const hasFallbackValue = keys.some((key) => fallbackValues[key] !== null && fallbackValues[key] !== undefined);
  if (timestamps.length === 0 && !hasFallbackValue) {
    return [];
  }

  if (timestamps.length === 0) {
    timestamps.push(Date.now() / 1000);
  }

  if (timestamps.length === 1) {
    timestamps.unshift(timestamps[0] - 2);
  }

  const rows = timestamps.map((timestamp) => {
    const row: Record<string, number | string> = {
      time: new Date(timestamp * 1000).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }),
    };
    for (const key of keys) {
      const fallbackValue = fallbackValues[key] ?? 0;
      const value = getValueForTimestamp(seriesByKey[key] ?? [], timestamp, fallbackValue);
      if (key === "kv_cache_usage" || key === "prefix_cache_hit_rate") {
        row[key] = value * 100;
      } else if (key === "cpu_prefix_cache_hit_rate") {
        row[key] = value * 100;
      } else if (key === "nixl_transfer_time_seconds") {
        row.nixl_transfer_time_ms = value * 1000;
      } else {
        row[key] = value;
      }
    }
    return row;
  });

  const firstLiveIndex = rows.findIndex((row) =>
    keys.some((key) => {
      const valueKey = key === "nixl_transfer_time_seconds" ? "nixl_transfer_time_ms" : key;
      return Math.abs(Number(row[valueKey] ?? 0)) > 1e-9;
    }),
  );

  if (firstLiveIndex <= 0) {
    return rows;
  }

  return rows.slice(firstLiveIndex - 1);
}

function percent(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function rate(value: number) {
  return formatDisplayNumber(value);
}

function integer(value: number) {
  return formatDisplayNumber(value);
}

function getValueForTimestamp(series: TimeSeriesPoint[], timestamp: number, fallbackValue: number) {
  let value = fallbackValue;
  for (const point of series) {
    if (point.timestamp > timestamp) {
      break;
    }
    value = point.value;
  }
  return value;
}

function formatDisplayNumber(value: number) {
  return Number.isFinite(value) ? value.toFixed(2) : "0.00";
}

function formatAxisNumber(value: number) {
  if (!Number.isFinite(value)) {
    return "0.00";
  }
  const absolute = Math.abs(value);
  if (absolute >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(2)}B`;
  }
  if (absolute >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(2)}M`;
  }
  if (absolute >= 1_000) {
    return `${(value / 1_000).toFixed(2)}k`;
  }
  return value.toFixed(2);
}

function formatPercentAxisNumber(value: number) {
  return `${formatDisplayNumber(value)}%`;
}

function formatTooltipValue(value: number | string | ReadonlyArray<number | string> | undefined) {
  const normalizedValue = Array.isArray(value) ? value[0] : value;
  return formatDisplayNumber(Number(normalizedValue ?? 0));
}

function formatPercentTooltipValue(value: number | string | ReadonlyArray<number | string> | undefined) {
  const normalizedValue = Array.isArray(value) ? value[0] : value;
  return `${formatDisplayNumber(Number(normalizedValue ?? 0))}%`;
}
