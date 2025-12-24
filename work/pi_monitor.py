#!/usr/bin/env python3
"""
Prime-RL Monitor - A Dolphie-inspired TUI for monitoring prime-rl training runs.

Features:
- Real-time GPU utilization for trainer and inference
- Throughput and performance graphs
- Live log streaming for orchestrator, trainer, and inference
- Training progress and metrics dashboard
"""

import asyncio
import re
import subprocess
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    RichLog,
    Rule,
    Static,
    TabbedContent,
    TabPane,
)

# Try to import pynvml for GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MonitorConfig:
    """Configuration for the Prime-RL monitor."""
    output_dir: Path = Path("outputs")
    refresh_interval: float = 1.0
    max_history: int = 120  # 2 minutes of history at 1s intervals
    trainer_gpu_ids: list[int] = field(default_factory=lambda: [0])
    inference_gpu_ids: list[int] = field(default_factory=lambda: [1])


# ============================================================================
# Data Collection
# ============================================================================

@dataclass
class GPUStats:
    """GPU statistics for a single GPU."""
    index: int
    name: str
    utilization: float  # 0-100
    memory_used: int  # MB
    memory_total: int  # MB
    temperature: int  # Celsius
    power_draw: float  # Watts


@dataclass 
class TrainingMetrics:
    """Parsed training metrics from logs."""
    step: int = 0
    total_steps: Optional[int] = None
    total_tokens: int = 0
    total_samples: int = 0
    throughput: float = 0.0
    reward_mean: float = 0.0
    async_level: int = 0
    ckpt_step: int = 0
    step_time: float = 0.0
    seq_len_mean: float = 0.0
    solve_all: float = 0.0
    solve_none: float = 0.0
    effective_batch_size: float = 0.0
    last_update: Optional[datetime] = None


class GPUCollector:
    """Collects GPU statistics using pynvml or nvidia-smi fallback."""
    
    def __init__(self):
        self.initialized = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                self.initialized = True
            except Exception:
                self.initialized = False
    
    def collect(self) -> list[GPUStats]:
        """Collect GPU stats for all available GPUs."""
        if self.initialized and PYNVML_AVAILABLE:
            return self._collect_pynvml()
        return self._collect_nvidia_smi()
    
    def _collect_pynvml(self) -> list[GPUStats]:
        """Collect using pynvml (faster)."""
        stats = []
        for i in range(self.device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except Exception:
                    power = 0.0
                
                stats.append(GPUStats(
                    index=i,
                    name=name,
                    utilization=util.gpu,
                    memory_used=mem.used // (1024 * 1024),
                    memory_total=mem.total // (1024 * 1024),
                    temperature=temp,
                    power_draw=power,
                ))
            except Exception:
                continue
        return stats
    
    def _collect_nvidia_smi(self) -> list[GPUStats]:
        """Fallback to nvidia-smi command."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return []
            
            stats = []
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:
                    stats.append(GPUStats(
                        index=int(parts[0]),
                        name=parts[1],
                        utilization=float(parts[2]) if parts[2] != '[N/A]' else 0,
                        memory_used=int(parts[3]) if parts[3] != '[N/A]' else 0,
                        memory_total=int(parts[4]) if parts[4] != '[N/A]' else 0,
                        temperature=int(parts[5]) if parts[5] != '[N/A]' else 0,
                        power_draw=float(parts[6]) if parts[6] != '[N/A]' else 0,
                    ))
            return stats
        except Exception:
            return []
    
    def close(self):
        """Cleanup pynvml."""
        if self.initialized and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


class LogParser:
    """Parses prime-rl log files to extract metrics."""
    
    # Regex patterns for parsing orchestrator logs
    STEP_PATTERN = re.compile(
        r'Step (\d+) \| Time: ([\d.]+)s \| Reward: ([\d.]+) \|.*?Throughput: ([\d.]+K?) tokens?/s \| '
        r'Seq\. Length: ([\d.]+).*?Async Level: (\d+)'
    )
    PROGRESS_PATTERN = re.compile(r"Starting orchestrator loop \(max_steps=(\d+|infinite)\)")
    TOKENS_PATTERN = re.compile(r'"progress/total_tokens":\s*(\d+)')
    SAMPLES_PATTERN = re.compile(r'"progress/total_samples":\s*(\d+)')
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.last_position = 0
        self.metrics = TrainingMetrics()
    
    def parse_new_lines(self) -> tuple[list[str], TrainingMetrics]:
        """Parse new lines from log file and return them with updated metrics."""
        new_lines = []
        
        if not self.log_path.exists():
            return new_lines, self.metrics
        
        try:
            with open(self.log_path, 'r') as f:
                f.seek(self.last_position)
                for line in f:
                    new_lines.append(line.rstrip())
                    self._parse_line(line)
                self.last_position = f.tell()
        except Exception:
            pass
        
        return new_lines, self.metrics
    
    def _parse_line(self, line: str):
        """Parse a single log line for metrics."""
        # Parse step summary lines
        match = self.STEP_PATTERN.search(line)
        if match:
            self.metrics.step = int(match.group(1))
            self.metrics.step_time = float(match.group(2))
            self.metrics.reward_mean = float(match.group(3))
            throughput_str = match.group(4)
            if 'K' in throughput_str:
                self.metrics.throughput = float(throughput_str.replace('K', '')) * 1000
            else:
                self.metrics.throughput = float(throughput_str)
            self.metrics.seq_len_mean = float(match.group(5))
            self.metrics.async_level = int(match.group(6))
            self.metrics.last_update = datetime.now()
            return
        
        # Parse max_steps
        match = self.PROGRESS_PATTERN.search(line)
        if match:
            max_steps_str = match.group(1)
            if max_steps_str != 'infinite':
                self.metrics.total_steps = int(max_steps_str)
            return
        
        # Parse total tokens
        match = self.TOKENS_PATTERN.search(line)
        if match:
            self.metrics.total_tokens = int(match.group(1))


class LogTailer:
    """Tails a log file and yields new lines."""
    
    def __init__(self, log_path: Path, max_lines: int = 1000):
        self.log_path = log_path
        self.max_lines = max_lines
        self.lines: Deque[str] = deque(maxlen=max_lines)
        self.last_position = 0
        self.last_size = 0
        self._load_existing()
    
    def _load_existing(self):
        """Load existing lines from the file."""
        if not self.log_path.exists():
            return
        
        try:
            with open(self.log_path, 'r', errors='replace') as f:
                # Read last N lines
                all_lines = f.readlines()
                for line in all_lines[-self.max_lines:]:
                    self.lines.append(line.rstrip())
                self.last_position = f.tell()
                self.last_size = self.log_path.stat().st_size
        except Exception:
            pass
    
    def get_new_lines(self) -> list[str]:
        """Get new lines since last check."""
        new_lines = []
        
        if not self.log_path.exists():
            return new_lines
        
        try:
            current_size = self.log_path.stat().st_size
            
            # Handle file truncation (e.g., log rotation)
            if current_size < self.last_size:
                self.last_position = 0
                self.lines.clear()
            
            self.last_size = current_size
            
            with open(self.log_path, 'r', errors='replace') as f:
                f.seek(self.last_position)
                for line in f:
                    stripped = line.rstrip()
                    if stripped:  # Skip empty lines
                        self.lines.append(stripped)
                        new_lines.append(stripped)
                self.last_position = f.tell()
        except Exception:
            pass
        
        return new_lines
    
    def get_all_lines(self) -> list[str]:
        """Get all buffered lines."""
        return list(self.lines)


# ============================================================================
# Widgets
# ============================================================================

class MetricCard(Static):
    """A card displaying a single metric with label and value."""
    
    DEFAULT_CSS = """
    MetricCard {
        width: 1fr;
        height: auto;
        min-height: 3;
        padding: 0 1;
        border: solid $primary-darken-2;
        background: $surface;
    }
    
    MetricCard .metric-label {
        color: $text-muted;
        text-style: dim;
    }
    
    MetricCard .metric-value {
        color: $text;
        text-style: bold;
    }
    """
    
    value = reactive("--")
    
    def __init__(self, label: str, value: str = "--", id: Optional[str] = None):
        super().__init__(id=id)
        self.label = label
        self.value = value
    
    def compose(self) -> ComposeResult:
        yield Label(self.label, classes="metric-label")
        yield Label(self.value, classes="metric-value", id="value")
    
    def watch_value(self, new_value: str):
        try:
            value_label = self.query_one("#value", Label)
            value_label.update(new_value)
        except Exception:
            pass


class GPUCard(Static):
    """A card showing GPU utilization with ASCII line graph."""
    
    DEFAULT_CSS = """
    GPUCard {
        width: 1fr;
        height: auto;
        min-height: 9;
        padding: 0 1;
        border: solid $primary-darken-2;
        background: $surface;
    }
    
    GPUCard .gpu-header {
        color: $text;
    }
    
    GPUCard .gpu-stats {
        color: $text-muted;
    }
    
    GPUCard .gpu-graph {
        height: 5;
    }
    """
    
    # Characters for drawing (bottom to top)
    CHARS = " ▁▂▃▄▅▆▇█"
    
    def __init__(self, label: str, gpu_ids: list[int], id: Optional[str] = None):
        super().__init__(id=id)
        self.label = label
        self.gpu_ids = gpu_ids
        self.history: Deque[float] = deque(maxlen=200)
    
    def compose(self) -> ComposeResult:
        ids_str = ",".join(str(i) for i in self.gpu_ids)
        yield Label(f"{self.label} [GPUs: {ids_str}]", classes="gpu-header")
        yield Static("", classes="gpu-graph", id="graph")
        yield Label("-- | --", classes="gpu-stats", id="stats")
    
    def _render_graph(self, width: int, height: int = 5) -> str:
        """Render as ASCII area chart."""
        if width < 10:
            width = 60
        
        # Reserve space for Y-axis
        graph_width = width - 6
        
        if not self.history:
            lines = []
            for i in range(height):
                pct = 100 - (i * 100 // height)
                lines.append(f"{pct:3d}% │")
            return "\n".join(lines)
        
        # Resample history to fit width
        data = list(self.history)
        if len(data) > graph_width:
            step = len(data) / graph_width
            data = [data[int(i * step)] for i in range(graph_width)]
        
        # Build graph rows (top to bottom: 100% -> 0%)
        lines = []
        for row in range(height):
            # Calculate threshold for this row
            row_min = 100 - ((row + 1) * 100 / height)  # e.g., row 0: 80-100, row 1: 60-80
            row_max = 100 - (row * 100 / height)
            pct_label = int(row_max)
            
            line = f"{pct_label:3d}% │"
            for val in data:
                if val >= row_max:
                    line += "█"
                elif val > row_min:
                    # Partial fill
                    frac = (val - row_min) / (row_max - row_min)
                    char_idx = int(frac * (len(self.CHARS) - 1))
                    line += self.CHARS[char_idx]
                else:
                    line += " "
            
            # Pad if data is shorter than width
            line += " " * (graph_width - len(data))
            lines.append(line)
        
        return "\n".join(lines)
    
    def update_stats(self, gpu_stats: list[GPUStats]):
        """Update with new GPU stats."""
        our_stats = [s for s in gpu_stats if s.index in self.gpu_ids]
        
        if not our_stats:
            return
        
        avg_util = sum(s.utilization for s in our_stats) / len(our_stats)
        total_mem_used = sum(s.memory_used for s in our_stats)
        total_mem_total = sum(s.memory_total for s in our_stats)
        avg_temp = sum(s.temperature for s in our_stats) / len(our_stats)
        
        self.history.append(avg_util)
        
        try:
            graph = self.query_one("#graph", Static)
            width = max(self.size.width - 4, 40)
            graph.update(self._render_graph(width, height=5))
            
            stats_label = self.query_one("#stats", Label)
            stats_label.update(
                f"{avg_util:.0f}% | {total_mem_used/1024:.1f}GB/{total_mem_total/1024:.1f}GB | {avg_temp:.0f}°C"
            )
        except Exception:
            pass


class ThroughputGraph(Static):
    """A line graph showing metrics over time."""
    
    DEFAULT_CSS = """
    ThroughputGraph {
        width: 1fr;
        height: auto;
        min-height: 5;
        padding: 0 1;
        border: solid $primary-darken-2;
        background: $surface;
    }
    
    ThroughputGraph .graph-label {
        color: $text-muted;
    }
    
    ThroughputGraph .graph-display {
        height: 3;
    }
    """
    
    CHARS = " ▁▂▃▄▅▆▇█"
    
    def __init__(self, label: str, unit: str = "", id: Optional[str] = None):
        super().__init__(id=id)
        self.label = label
        self.unit = unit
        self.history: Deque[float] = deque(maxlen=200)
    
    def compose(self) -> ComposeResult:
        yield Label(f"{self.label}: --{self.unit}", classes="graph-label", id="label")
        yield Static("", classes="graph-display", id="graph")
    
    def _render_graph(self, width: int, height: int = 3) -> str:
        """Render as simple ASCII sparkline-style graph."""
        if not self.history or width < 5:
            return "\n" * height
        
        data = list(self.history)
        
        # Auto-scale based on data range
        min_val = min(data)
        max_val = max(data)
        if max_val == min_val:
            max_val = min_val + 1
        
        # Resample to fit width
        if len(data) > width:
            step = len(data) / width
            data = [data[int(i * step)] for i in range(width)]
        
        # Normalize to 0-1
        normalized = [(v - min_val) / (max_val - min_val) for v in data]
        
        # Build rows
        lines = []
        for row in range(height):
            row_min = 1.0 - ((row + 1) / height)
            row_max = 1.0 - (row / height)
            
            line = ""
            for val in normalized:
                if val >= row_max:
                    line += "█"
                elif val > row_min:
                    frac = (val - row_min) / (row_max - row_min)
                    char_idx = int(frac * (len(self.CHARS) - 1))
                    line += self.CHARS[char_idx]
                else:
                    line += " "
            
            line += " " * (width - len(data))
            lines.append(line)
        
        return "\n".join(lines)
    
    def add_value(self, value: float):
        """Add a new value to the history."""
        self.history.append(value)
        
        try:
            label = self.query_one("#label", Label)
            if value >= 1000000:
                label.update(f"{self.label}: {value/1000000:.2f}M{self.unit}")
            elif value >= 1000:
                label.update(f"{self.label}: {value/1000:.1f}K{self.unit}")
            else:
                label.update(f"{self.label}: {value:.2f}{self.unit}")
            
            graph = self.query_one("#graph", Static)
            width = max(self.size.width - 4, 30)
            graph.update(self._render_graph(width, height=3))
        except Exception:
            pass


# ============================================================================
# Panels
# ============================================================================

class DashboardPanel(Static):
    """Main dashboard panel showing key metrics."""
    
    DEFAULT_CSS = """
    DashboardPanel {
        width: 100%;
        height: auto;
        padding: 1;
    }
    
    DashboardPanel Horizontal {
        height: auto;
    }
    """
    
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield MetricCard("Step", id="metric-step")
            yield MetricCard("Throughput", id="metric-throughput")
            yield MetricCard("Reward", id="metric-reward")
            yield MetricCard("Async Level", id="metric-async")
        with Horizontal():
            yield MetricCard("Total Tokens", id="metric-tokens")
            yield MetricCard("Seq Length", id="metric-seqlen")
            yield MetricCard("Step Time", id="metric-steptime")
            yield MetricCard("Checkpoint", id="metric-ckpt")
    
    def update_metrics(self, metrics: TrainingMetrics):
        """Update dashboard with new metrics."""
        try:
            # Step
            step_card = self.query_one("#metric-step", MetricCard)
            if metrics.total_steps:
                step_card.value = f"{metrics.step}/{metrics.total_steps}"
            else:
                step_card.value = str(metrics.step)
            
            # Throughput
            tp_card = self.query_one("#metric-throughput", MetricCard)
            if metrics.throughput >= 1000:
                tp_card.value = f"{metrics.throughput/1000:.1f}K tok/s"
            else:
                tp_card.value = f"{metrics.throughput:.0f} tok/s"
            
            # Reward
            reward_card = self.query_one("#metric-reward", MetricCard)
            reward_card.value = f"{metrics.reward_mean:.4f}"
            
            # Async level
            async_card = self.query_one("#metric-async", MetricCard)
            async_card.value = str(metrics.async_level)
            
            # Total tokens
            tokens_card = self.query_one("#metric-tokens", MetricCard)
            if metrics.total_tokens >= 1_000_000_000:
                tokens_card.value = f"{metrics.total_tokens/1_000_000_000:.2f}B"
            elif metrics.total_tokens >= 1_000_000:
                tokens_card.value = f"{metrics.total_tokens/1_000_000:.2f}M"
            elif metrics.total_tokens >= 1_000:
                tokens_card.value = f"{metrics.total_tokens/1_000:.1f}K"
            else:
                tokens_card.value = str(metrics.total_tokens)
            
            # Seq length
            seqlen_card = self.query_one("#metric-seqlen", MetricCard)
            seqlen_card.value = f"{metrics.seq_len_mean:.0f}"
            
            # Step time
            steptime_card = self.query_one("#metric-steptime", MetricCard)
            steptime_card.value = f"{metrics.step_time:.2f}s"
            
            # Checkpoint
            ckpt_card = self.query_one("#metric-ckpt", MetricCard)
            ckpt_card.value = str(metrics.ckpt_step)
        except Exception:
            pass


class GPUPanel(Static):
    """Panel showing GPU utilization for trainer and inference."""
    
    DEFAULT_CSS = """
    GPUPanel {
        width: 100%;
        height: auto;
        padding: 1;
    }
    
    GPUPanel Horizontal {
        height: auto;
    }
    """
    
    def __init__(self, trainer_gpus: list[int], inference_gpus: list[int], id: Optional[str] = None):
        super().__init__(id=id)
        self.trainer_gpus = trainer_gpus
        self.inference_gpus = inference_gpus
    
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield GPUCard("Trainer", self.trainer_gpus, id="gpu-trainer")
            yield GPUCard("Inference", self.inference_gpus, id="gpu-inference")
    
    def update_stats(self, gpu_stats: list[GPUStats]):
        """Update GPU stats."""
        try:
            self.query_one("#gpu-trainer", GPUCard).update_stats(gpu_stats)
            self.query_one("#gpu-inference", GPUCard).update_stats(gpu_stats)
        except Exception:
            pass


class GraphsPanel(Static):
    """Panel showing throughput and performance graphs."""
    
    DEFAULT_CSS = """
    GraphsPanel {
        width: 100%;
        height: auto;
        padding: 1;
    }
    
    GraphsPanel Horizontal {
        height: auto;
    }
    """
    
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield ThroughputGraph("Throughput", " tok/s", id="graph-throughput")
            yield ThroughputGraph("Step Time", "s", id="graph-steptime")
        with Horizontal():
            yield ThroughputGraph("Reward", "", id="graph-reward")
            yield ThroughputGraph("Seq Length", " tokens", id="graph-seqlen")
    
    def update_metrics(self, metrics: TrainingMetrics):
        """Update graphs with new metrics."""
        if metrics.throughput > 0:
            try:
                self.query_one("#graph-throughput", ThroughputGraph).add_value(metrics.throughput)
            except Exception:
                pass
        
        if metrics.step_time > 0:
            try:
                self.query_one("#graph-steptime", ThroughputGraph).add_value(metrics.step_time)
            except Exception:
                pass
        
        if metrics.reward_mean > 0:
            try:
                self.query_one("#graph-reward", ThroughputGraph).add_value(metrics.reward_mean)
            except Exception:
                pass
        
        if metrics.seq_len_mean > 0:
            try:
                self.query_one("#graph-seqlen", ThroughputGraph).add_value(metrics.seq_len_mean)
            except Exception:
                pass


class LogsPanel(Static):
    """Panel showing logs from all three components with tabs."""
    
    DEFAULT_CSS = """
    LogsPanel {
        width: 100%;
        height: 1fr;
        min-height: 15;
    }
    
    LogsPanel TabbedContent {
        height: 100%;
    }
    
    LogsPanel TabPane {
        height: 100%;
        padding: 0;
    }
    
    LogsPanel RichLog {
        height: 100%;
        min-height: 12;
        background: $surface;
        border: solid $primary-darken-3;
    }
    """
    
    def compose(self) -> ComposeResult:
        with TabbedContent():
            with TabPane("Orchestrator", id="tab-orch"):
                yield RichLog(id="log-orch", wrap=True, highlight=True, markup=True, auto_scroll=True, max_lines=500)
            with TabPane("Trainer", id="tab-trainer"):
                yield RichLog(id="log-trainer", wrap=True, highlight=True, markup=True, auto_scroll=True, max_lines=500)
            with TabPane("Inference", id="tab-inference"):
                yield RichLog(id="log-inference", wrap=True, highlight=True, markup=True, auto_scroll=True, max_lines=500)
    
    def add_log_line(self, component: str, line: str):
        """Add a log line to the appropriate log view."""
        log_id = f"log-{component}"
        try:
            log = self.query_one(f"#{log_id}", RichLog)
            styled_line = self._style_log_line(line)
            log.write(styled_line)
        except Exception:
            pass
    
    def _style_log_line(self, line: str) -> Text:
        """Apply styling to log line based on content."""
        text = Text(line)
        
        # Color based on log level
        if 'ERROR' in line or 'error' in line.lower():
            text.stylize("red")
        elif 'WARNING' in line or 'warning' in line.lower():
            text.stylize("yellow")
        elif 'SUCCESS' in line or '✓' in line:
            text.stylize("green")
        elif 'DEBUG' in line:
            text.stylize("dim")
        elif 'INFO' in line:
            text.stylize("blue")
        
        return text
    
    def load_initial_logs(self, component: str, lines: list[str]):
        """Load initial log lines for a component."""
        import sys
        log_id = f"log-{component}"
        try:
            log = self.query_one(f"#{log_id}", RichLog)
            print(f"[PrimeMonitor] load_initial_logs: writing {len(lines[-100:])} lines to {log_id}", file=sys.stderr)
            for line in lines[-100:]:  # Last 100 lines
                styled_line = self._style_log_line(line)
                log.write(styled_line)
        except Exception as e:
            print(f"[PrimeMonitor] load_initial_logs ERROR: {e}", file=sys.stderr)


# ============================================================================
# Main Application
# ============================================================================

class PrimeMonitor(App):
    """Main Prime-RL Monitor application."""
    
    CSS = """
    Screen {
        background: $background;
    }
    
    #main-container {
        width: 100%;
        height: 100%;
        padding: 0 1;
    }
    
    .section-title {
        color: $primary;
        text-style: bold;
        padding: 0 1;
        margin-top: 1;
    }
    
    #status-bar {
        dock: bottom;
        height: 1;
        background: $primary-darken-3;
        color: $text;
        padding: 0 1;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("p", "toggle_pause", "Pause"),
        Binding("1", "focus_orch", "Orchestrator"),
        Binding("2", "focus_trainer", "Trainer"),
        Binding("3", "focus_inference", "Inference"),
    ]
    
    paused = reactive(False)
    
    def __init__(self, config: MonitorConfig):
        super().__init__()
        self.config = config
        self.gpu_collector = GPUCollector()
        
        # Set up log paths - try multiple locations
        self.run_dir = self._find_run_dir()
        
        # Possible log locations (in priority order)
        orch_paths = [
            self.config.output_dir / "logs" / "orchestrator.stdout",  # From rl.py redirect
            self.run_dir / "logs" / "orchestrator.log" if self.run_dir else None,
            self.config.output_dir / "run_default" / "logs" / "orchestrator.log",
        ]
        trainer_paths = [
            self.config.output_dir / "logs" / "trainer.stdout",
            self.config.output_dir / "torchrun" / "0" / "stdout.log",
        ]
        inference_paths = [
            self.config.output_dir / "logs" / "inference.stdout",
        ]
        
        def find_log(paths: list) -> Optional[Path]:
            for p in paths:
                if p and p.exists():
                    return p
            # Return first non-None path (will be created when training starts)
            for p in paths:
                if p:
                    return p
            return None
        
        self.log_paths = {
            "orch": find_log(orch_paths),
            "trainer": find_log(trainer_paths),
            "inference": find_log(inference_paths),
        }
        
        # Debug: Print discovered log paths to stderr
        import sys
        print(f"[PrimeMonitor] Log paths discovered:", file=sys.stderr)
        for k, v in self.log_paths.items():
            exists = v.exists() if v else False
            print(f"  {k}: {v} (exists={exists})", file=sys.stderr)
        
        # Initialize log tailers
        self.log_tailers = {}
        for component, path in self.log_paths.items():
            if path:
                self.log_tailers[component] = LogTailer(path)
        
        # Initialize metrics parser (uses orchestrator log)
        if self.log_paths["orch"]:
            self.metrics_parser = LogParser(self.log_paths["orch"])
        else:
            self.metrics_parser = None
        
        self.metrics = TrainingMetrics()
    
    def _find_run_dir(self) -> Optional[Path]:
        """Find the run directory (run_default or run_*)."""
        # Check for run_default first
        run_default = self.config.output_dir / "run_default"
        if run_default.exists():
            return run_default
        
        # Look for any run_* directory
        run_dirs = list(self.config.output_dir.glob("run_*"))
        if run_dirs:
            # Return the most recently modified
            return max(run_dirs, key=lambda p: p.stat().st_mtime if p.exists() else 0)
        
        return None
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with VerticalScroll(id="main-container"):
            yield Label("📊 Dashboard", classes="section-title")
            yield DashboardPanel(id="dashboard")
            
            yield Label("🖥️ GPU Utilization", classes="section-title")
            yield GPUPanel(
                self.config.trainer_gpu_ids,
                self.config.inference_gpu_ids,
                id="gpu-panel"
            )
            
            yield Label("📈 Performance", classes="section-title")
            yield GraphsPanel(id="graphs")
            
            yield Label("📜 Logs", classes="section-title")
            yield LogsPanel(id="logs")
        
        yield Static(
            f"Output: {self.config.output_dir} | Refresh: {self.config.refresh_interval}s | [P]ause [Q]uit",
            id="status-bar"
        )
        yield Footer()
    
    def on_mount(self):
        """Initialize on mount."""
        import sys
        print(f"[PrimeMonitor] on_mount called", file=sys.stderr)
        
        # Use call_after_refresh to ensure widgets are ready
        self.call_after_refresh(self._load_initial_logs)
        
        # Start refresh timer
        self.set_interval(self.config.refresh_interval, self.refresh_data)
    
    def _load_initial_logs(self):
        """Load initial logs after widgets are ready."""
        import sys
        print(f"[PrimeMonitor] _load_initial_logs called", file=sys.stderr)
        
        try:
            logs_panel = self.query_one("#logs", LogsPanel)
            
            for component, tailer in self.log_tailers.items():
                path = self.log_paths.get(component)
                if path:
                    exists = path.exists()
                    status = "✓ found" if exists else "⏳ waiting"
                    logs_panel.add_log_line(component, f"[{status}] {path}")
                    
                    if exists:
                        initial_lines = tailer.get_all_lines()
                        print(f"[PrimeMonitor] {component}: {len(initial_lines)} lines", file=sys.stderr)
                        logs_panel.load_initial_logs(component, initial_lines)
                else:
                    logs_panel.add_log_line(component, "[no log path configured]")
            
            # Initial data refresh
            self.refresh_data()
            
        except Exception as e:
            print(f"[PrimeMonitor] _load_initial_logs ERROR: {e}", file=sys.stderr)
    
    def refresh_data(self):
        """Refresh all data."""
        if self.paused:
            return
        
        # Collect GPU stats
        gpu_stats = self.gpu_collector.collect()
        if gpu_stats:
            try:
                self.query_one("#gpu-panel", GPUPanel).update_stats(gpu_stats)
            except Exception:
                pass
        
        # Parse new log lines and update metrics
        if self.metrics_parser:
            new_lines, self.metrics = self.metrics_parser.parse_new_lines()
            
            try:
                self.query_one("#dashboard", DashboardPanel).update_metrics(self.metrics)
                self.query_one("#graphs", GraphsPanel).update_metrics(self.metrics)
            except Exception:
                pass
        
        # Update log views
        logs_panel = self.query_one("#logs", LogsPanel)
        for component, tailer in self.log_tailers.items():
            new_lines = tailer.get_new_lines()
            for line in new_lines:
                logs_panel.add_log_line(component, line)
        
        # Update title with current step
        if self.metrics.step > 0:
            if self.metrics.total_steps:
                self.title = f"Prime-RL Monitor | Step {self.metrics.step}/{self.metrics.total_steps}"
            else:
                self.title = f"Prime-RL Monitor | Step {self.metrics.step}"
    
    def action_refresh(self):
        """Manual refresh."""
        self.refresh_data()
    
    def action_toggle_pause(self):
        """Toggle pause state."""
        self.paused = not self.paused
        status = self.query_one("#status-bar", Static)
        if self.paused:
            status.update(f"⏸️  PAUSED | Output: {self.config.output_dir} | [P] Resume [Q]uit")
        else:
            status.update(
                f"Output: {self.config.output_dir} | Refresh: {self.config.refresh_interval}s | [P]ause [Q]uit"
            )
    
    def action_focus_orch(self):
        """Switch to orchestrator logs tab."""
        try:
            tabbed = self.query_one(TabbedContent)
            tabbed.active = "tab-orch"
        except Exception:
            pass
    
    def action_focus_trainer(self):
        """Switch to trainer logs tab."""
        try:
            tabbed = self.query_one(TabbedContent)
            tabbed.active = "tab-trainer"
        except Exception:
            pass
    
    def action_focus_inference(self):
        """Switch to inference logs tab."""
        try:
            tabbed = self.query_one(TabbedContent)
            tabbed.active = "tab-inference"
        except Exception:
            pass
    
    def on_unmount(self):
        """Cleanup on exit."""
        self.gpu_collector.close()


# ============================================================================
# Entry Point
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prime-RL Monitor - TUI for monitoring prime-rl training")
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to the outputs directory (default: outputs)"
    )
    parser.add_argument(
        "-r", "--refresh",
        type=float,
        default=1.0,
        help="Refresh interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--trainer-gpus",
        type=str,
        default="0",
        help="Comma-separated list of trainer GPU IDs (default: 0)"
    )
    parser.add_argument(
        "--inference-gpus",
        type=str,
        default="1",
        help="Comma-separated list of inference GPU IDs (default: 1)"
    )
    
    args = parser.parse_args()
    
    config = MonitorConfig(
        output_dir=args.output_dir,
        refresh_interval=args.refresh,
        trainer_gpu_ids=[int(x) for x in args.trainer_gpus.split(",")],
        inference_gpu_ids=[int(x) for x in args.inference_gpus.split(",")],
    )
    
    # Debug output
    import sys
    print(f"[PrimeMonitor] Config:", file=sys.stderr)
    print(f"  output_dir: {config.output_dir}", file=sys.stderr)
    print(f"  trainer_gpus: {config.trainer_gpu_ids}", file=sys.stderr)
    print(f"  inference_gpus: {config.inference_gpu_ids}", file=sys.stderr)
    
    app = PrimeMonitor(config)
    app.run()


if __name__ == "__main__":
    main()