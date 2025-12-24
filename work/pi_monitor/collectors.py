# prime_monitor/collectors.py
"""Data collectors for GPU and system metrics."""

import subprocess
from typing import Optional

from .config import GPUStats

# Try to import pynvml for GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class GPUCollector:
    """Collects GPU statistics using pynvml or nvidia-smi fallback."""
    
    def __init__(self):
        self.use_pynvml = PYNVML_AVAILABLE
        self._initialized = False
        
        if self.use_pynvml:
            try:
                pynvml.nvmlInit()
                self._initialized = True
            except Exception:
                self.use_pynvml = False
    
    def collect(self) -> list[GPUStats]:
        """Collect stats from all GPUs."""
        if self.use_pynvml and self._initialized:
            return self._collect_pynvml()
        return self._collect_nvidia_smi()
    
    def _collect_pynvml(self) -> list[GPUStats]:
        """Collect using pynvml (faster)."""
        stats = []
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
                except Exception:
                    power = 0.0
                
                stats.append(GPUStats(
                    index=i,
                    name=name,
                    utilization=util.gpu,
                    memory_used=memory.used / 1024 / 1024,  # bytes to MB
                    memory_total=memory.total / 1024 / 1024,
                    temperature=temp,
                    power_draw=power,
                ))
        except Exception:
            pass
        return stats
    
    def _collect_nvidia_smi(self) -> list[GPUStats]:
        """Collect using nvidia-smi (slower fallback)."""
        stats = []
        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 7:
                        stats.append(GPUStats(
                            index=int(parts[0]),
                            name=parts[1],
                            utilization=float(parts[2]) if parts[2] not in ['[N/A]', ''] else 0.0,
                            memory_used=float(parts[3]) if parts[3] not in ['[N/A]', ''] else 0.0,
                            memory_total=float(parts[4]) if parts[4] not in ['[N/A]', ''] else 0.0,
                            temperature=float(parts[5]) if parts[5] not in ['[N/A]', ''] else 0.0,
                            power_draw=float(parts[6]) if parts[6] not in ['[N/A]', ''] else 0.0,
                        ))
        except Exception:
            pass
        return stats
    
    def close(self):
        """Cleanup pynvml."""
        if self.use_pynvml and self._initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass