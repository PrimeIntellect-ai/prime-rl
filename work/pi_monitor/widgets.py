# prime_monitor/widgets.py
"""UI Widgets for Prime Monitor."""

from collections import deque
from typing import Deque, Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import Label, Static

from .config import GPUStats


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
    }
    
    MetricCard .metric-value {
        color: $text;
        text-style: bold;
    }
    """
    
    def __init__(self, label: str, id: Optional[str] = None):
        super().__init__(id=id)
        self.label_text = label
        self._value = "--"
    
    def compose(self) -> ComposeResult:
        yield Label(self.label_text, classes="metric-label")
        yield Label(self._value, classes="metric-value", id="value")
    
    @property
    def value(self) -> str:
        return self._value
    
    @value.setter
    def value(self, new_value: str):
        self._value = new_value
        try:
            self.query_one("#value", Label).update(new_value)
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
    
    # Characters for drawing (increasing fill)
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
            row_min = 100 - ((row + 1) * 100 / height)
            row_max = 100 - (row * 100 / height)
            pct_label = int(row_max)
            
            line = f"{pct_label:3d}% │"
            for val in data:
                if val >= row_max:
                    line += "█"
                elif val > row_min:
                    frac = (val - row_min) / (row_max - row_min)
                    char_idx = int(frac * (len(self.CHARS) - 1))
                    line += self.CHARS[char_idx]
                else:
                    line += " "
            
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