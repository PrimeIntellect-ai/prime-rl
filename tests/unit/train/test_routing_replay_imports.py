"""Keep tensor metadata/loader imports independent of optional model kernels."""

import subprocess
import sys


def test_routing_data_import_does_not_load_model_registry():
    code = """
import sys
from prime_rl.trainer.rl.data import DataLoader
from prime_rl.trainer.routing_replay import RoutingReplay
for name in ("prime_rl.trainer.models", "ring_flash_attn", "flash_attn"):
    assert name not in sys.modules, name
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
