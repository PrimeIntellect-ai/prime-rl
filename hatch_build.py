"""Hatchling build hook to install .pth file for editable installs."""

import shutil
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Build hook to ensure .pth file is installed for editable installs."""

    def initialize(self, version, build_data):
        """Install .pth file to site-packages for editable installs."""
        if self.target_name == "editable":
            # Find site-packages directory
            for site_path in sys.path:
                site_path = Path(site_path)
                if site_path.name == "site-packages" and site_path.exists():
                    # Copy .pth file
                    pth_source = Path(self.root) / "src/prime_rl/data/prime_rl_vllm_patch.pth"
                    pth_dest = site_path / "prime_rl_vllm_patch.pth"

                    if pth_source.exists():
                        shutil.copy2(pth_source, pth_dest)
                        print(f"✓ Installed {pth_dest}")
                    break
