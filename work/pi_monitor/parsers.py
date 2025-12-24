# prime_monitor/parsers.py
"""Log parsing utilities for extracting metrics."""

import re
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional

from .config import TrainingMetrics


class LogParser:
    """Parses orchestrator logs to extract training metrics."""
    
    # Regex patterns for different log formats
    # Pattern 1: "Step 42 | Time: 4.2s | Reward: 0.847 | Throughput: 12.4K tokens/s | Seq. Length: 1500 | Async Level: 2"
    STEP_PATTERN = re.compile(
        r'Step\s+(\d+)\s*\|\s*Time:\s*([\d.]+)s?\s*\|\s*Reward:\s*([\d.]+)\s*\|\s*Throughput:\s*([\d.]+)K?\s*tokens?/s\s*\|\s*Seq\.?\s*Length:\s*([\d.]+)\s*\|\s*Async\s*Level:\s*(\d+)',
        re.IGNORECASE
    )
    
    # Pattern 2: Simpler format "step=42 reward=0.847 throughput=12400"
    SIMPLE_PATTERN = re.compile(
        r'step[=:\s]+(\d+).*?reward[=:\s]+([\d.]+).*?throughput[=:\s]+([\d.]+)',
        re.IGNORECASE
    )
    
    # Pattern for progress/max steps
    PROGRESS_PATTERN = re.compile(r'max_steps[=:\s]+(\d+|infinite)', re.IGNORECASE)
    
    # Pattern for total tokens
    TOKENS_PATTERN = re.compile(r'total[_\s]?tokens[=:\s]+(\d+)', re.IGNORECASE)
    
    # Pattern for checkpoint
    CKPT_PATTERN = re.compile(r'checkpoint.*?step[=:\s]+(\d+)', re.IGNORECASE)
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.last_position = 0
        self.metrics = TrainingMetrics()
        self._seek_to_end()
    
    def _seek_to_end(self):
        """Start from end of file to avoid processing old logs."""
        if self.log_path.exists():
            try:
                self.last_position = self.log_path.stat().st_size
            except Exception:
                pass
    
    def parse_new_lines(self) -> tuple[list[str], TrainingMetrics]:
        """Parse new lines from log file and return them with updated metrics."""
        new_lines = []
        
        if not self.log_path.exists():
            return new_lines, self.metrics
        
        try:
            with open(self.log_path, 'r', errors='replace') as f:
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
        # Try main step pattern first
        match = self.STEP_PATTERN.search(line)
        if match:
            self.metrics.step = int(match.group(1))
            self.metrics.step_time = float(match.group(2))
            self.metrics.reward_mean = float(match.group(3))
            throughput_str = match.group(4)
            # Handle K suffix
            if 'K' in line[match.start(4):match.end(4)+2].upper():
                self.metrics.throughput = float(throughput_str) * 1000
            else:
                self.metrics.throughput = float(throughput_str)
            self.metrics.seq_len_mean = float(match.group(5))
            self.metrics.async_level = int(match.group(6))
            self.metrics.last_update = datetime.now().strftime("%H:%M:%S")
            return
        
        # Try simpler pattern
        match = self.SIMPLE_PATTERN.search(line)
        if match:
            self.metrics.step = int(match.group(1))
            self.metrics.reward_mean = float(match.group(2))
            self.metrics.throughput = float(match.group(3))
            self.metrics.last_update = datetime.now().strftime("%H:%M:%S")
            return
        
        # Parse max_steps
        match = self.PROGRESS_PATTERN.search(line)
        if match:
            max_steps_str = match.group(1)
            if max_steps_str.lower() != 'infinite':
                self.metrics.total_steps = int(max_steps_str)
            return
        
        # Parse total tokens
        match = self.TOKENS_PATTERN.search(line)
        if match:
            self.metrics.total_tokens = int(match.group(1))
        
        # Parse checkpoint
        match = self.CKPT_PATTERN.search(line)
        if match:
            self.metrics.ckpt_step = int(match.group(1))


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
                    stripped = line.rstrip()
                    if stripped:
                        self.lines.append(stripped)
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
                    if stripped:
                        self.lines.append(stripped)
                        new_lines.append(stripped)
                self.last_position = f.tell()
        except Exception:
            pass
        
        return new_lines
    
    def get_all_lines(self) -> list[str]:
        """Get all buffered lines."""
        return list(self.lines)