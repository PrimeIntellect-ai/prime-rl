# prime_monitor/app.py
"""Main application for Prime Monitor."""

import sys
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widgets import Footer, Header, Label, Static, TabbedContent

from .collectors import GPUCollector
from .config import MonitorConfig, TrainingMetrics
from .panels import DashboardPanel, GPUPanel, GraphsPanel, LogsPanel
from .parsers import LogParser, LogTailer


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
        color: $text-muted;
        padding: 0 1;
    }
    """
    
    TITLE = "PrimeMonitor"
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("p", "toggle_pause", "Pause"),
        Binding("r", "refresh", "Refresh"),
        Binding("1", "focus_orch", "Orchestrator"),
        Binding("2", "focus_trainer", "Trainer"),
        Binding("3", "focus_inference", "Inference"),
    ]
    
    paused = reactive(False)
    
    def __init__(self, config: MonitorConfig):
        super().__init__()
        self.config = config
        self.gpu_collector = GPUCollector()
        
        # Set up log paths
        self.run_dir = self._find_run_dir()
        
        # Possible log locations (in priority order)
        orch_paths = [
            self.config.output_dir / "logs" / "orchestrator.stdout",
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
            for p in paths:
                if p:
                    return p
            return None
        
        self.log_paths = {
            "orch": find_log(orch_paths),
            "trainer": find_log(trainer_paths),
            "inference": find_log(inference_paths),
        }
        
        # Debug output
        print(f"[PrimeMonitor] Log paths:", file=sys.stderr)
        for k, v in self.log_paths.items():
            exists = v.exists() if v else False
            print(f"  {k}: {v} (exists={exists})", file=sys.stderr)
        
        # Initialize log tailers
        self.log_tailers = {}
        for component, path in self.log_paths.items():
            if path:
                self.log_tailers[component] = LogTailer(path)
        
        # Initialize metrics parser
        if self.log_paths["orch"]:
            self.metrics_parser = LogParser(self.log_paths["orch"])
        else:
            self.metrics_parser = None
        
        self.metrics = TrainingMetrics()
    
    def _find_run_dir(self) -> Optional[Path]:
        """Find the run directory."""
        run_default = self.config.output_dir / "run_default"
        if run_default.exists():
            return run_default
        
        run_dirs = list(self.config.output_dir.glob("run_*"))
        if run_dirs:
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
        print(f"[PrimeMonitor] on_mount called", file=sys.stderr)
        self.call_after_refresh(self._load_initial_logs)
        self.set_interval(self.config.refresh_interval, self.refresh_data)
    
    def _load_initial_logs(self):
        """Load initial logs after widgets are ready."""
        print(f"[PrimeMonitor] _load_initial_logs called", file=sys.stderr)
        
        try:
            logs_panel = self.query_one("#logs", LogsPanel)
            
            for component, tailer in self.log_tailers.items():
                path = self.log_paths.get(component)
                if path:
                    exists = path.exists()
                    status = "✓" if exists else "⏳"
                    logs_panel.add_log_line(component, f"[{status}] {path}")
                    
                    if exists:
                        initial_lines = tailer.get_all_lines()
                        print(f"[PrimeMonitor] {component}: {len(initial_lines)} lines", file=sys.stderr)
                        logs_panel.load_initial_logs(component, initial_lines)
                else:
                    logs_panel.add_log_line(component, "[no log path]")
            
            self.refresh_data()
            
        except Exception as e:
            print(f"[PrimeMonitor] _load_initial_logs ERROR: {e}", file=sys.stderr)
    
    def refresh_data(self):
        """Refresh all data."""
        if self.paused:
            return
        
        # GPU stats
        gpu_stats = self.gpu_collector.collect()
        if gpu_stats:
            try:
                self.query_one("#gpu-panel", GPUPanel).update_stats(gpu_stats)
            except Exception:
                pass
        
        # Metrics from logs
        if self.metrics_parser:
            new_lines, self.metrics = self.metrics_parser.parse_new_lines()
            
            try:
                self.query_one("#dashboard", DashboardPanel).update_metrics(self.metrics)
                self.query_one("#graphs", GraphsPanel).update_metrics(self.metrics)
            except Exception:
                pass
        
        # Log updates
        try:
            logs_panel = self.query_one("#logs", LogsPanel)
            for component, tailer in self.log_tailers.items():
                new_lines = tailer.get_new_lines()
                for line in new_lines:
                    logs_panel.add_log_line(component, line)
        except Exception:
            pass
        
        # Update title
        if self.metrics.step > 0:
            if self.metrics.total_steps:
                self.title = f"Prime-RL Monitor | Step {self.metrics.step}/{self.metrics.total_steps}"
            else:
                self.title = f"Prime-RL Monitor | Step {self.metrics.step}"
    
    def action_refresh(self):
        self.refresh_data()
    
    def action_toggle_pause(self):
        self.paused = not self.paused
        try:
            status = self.query_one("#status-bar", Static)
            if self.paused:
                status.update(f"⏸️ PAUSED | Output: {self.config.output_dir} | [P] Resume [Q]uit")
            else:
                status.update(
                    f"Output: {self.config.output_dir} | Refresh: {self.config.refresh_interval}s | [P]ause [Q]uit"
                )
        except Exception:
            pass
    
    def action_focus_orch(self):
        try:
            tabbed = self.query_one(TabbedContent)
            tabbed.active = "tab-orch"
        except Exception:
            pass
    
    def action_focus_trainer(self):
        try:
            tabbed = self.query_one(TabbedContent)
            tabbed.active = "tab-trainer"
        except Exception:
            pass
    
    def action_focus_inference(self):
        try:
            tabbed = self.query_one(TabbedContent)
            tabbed.active = "tab-inference"
        except Exception:
            pass
    
    def on_unmount(self):
        self.gpu_collector.close()