# prime_monitor/__init__.py
"""Prime Monitor - A TUI for monitoring prime-rl training runs."""

from .app import PrimeMonitor
from .config import MonitorConfig, GPUStats, TrainingMetrics
from .collectors import GPUCollector
from .parsers import LogParser, LogTailer
from .widgets import MetricCard, GPUCard, ThroughputGraph
from .panels import DashboardPanel, GPUPanel, GraphsPanel, LogsPanel

__version__ = "0.2.0"
__all__ = [
    "PrimeMonitor",
    "MonitorConfig",
    "GPUStats",
    "TrainingMetrics",
    "GPUCollector",
    "LogParser",
    "LogTailer",
    "MetricCard",
    "GPUCard",
    "ThroughputGraph",
    "DashboardPanel",
    "GPUPanel",
    "GraphsPanel",
    "LogsPanel",
]