# prime_monitor/panels.py
"""Panel containers for Prime Monitor."""

from typing import Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Label, RichLog, Static, TabbedContent, TabPane

from .config import GPUStats, TrainingMetrics
from .widgets import GPUCard, MetricCard, ThroughputGraph


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
        
        lower = line.lower()
        if 'error' in lower:
            text.stylize("red")
        elif 'warning' in lower or 'warn' in lower:
            text.stylize("yellow")
        elif 'success' in lower or '✓' in line:
            text.stylize("green")
        elif 'debug' in lower:
            text.stylize("dim")
        elif 'info' in lower:
            text.stylize("blue")
        
        return text
    
    def load_initial_logs(self, component: str, lines: list[str]):
        """Load initial log lines for a component."""
        log_id = f"log-{component}"
        try:
            log = self.query_one(f"#{log_id}", RichLog)
            for line in lines[-100:]:  # Last 100 lines
                styled_line = self._style_log_line(line)
                log.write(styled_line)
        except Exception:
            pass