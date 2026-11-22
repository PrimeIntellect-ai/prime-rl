from __future__ import annotations

import subprocess
import sys
import textwrap


def test_base_package_import_does_not_require_craftax_fork() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockCraftax(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "craftax" or fullname.startswith("craftax."):
                    raise ModuleNotFoundError(
                        "simulated base install without Craftax", name="craftax"
                    )
                return None

        sys.meta_path.insert(0, BlockCraftax())

        import glyphbench
        from glyphbench.core import list_task_ids, make_env

        task_ids = list_task_ids()
        assert "glyphbench/craftaxfull-v0" not in task_ids
        assert "glyphbench/craftax-choptrees-v0" in task_ids
        env = make_env("glyphbench/minigrid-empty-5x5-v0")
        _observation, info = env.reset(42)
        assert info["env_id"] == "glyphbench/minigrid-empty-5x5-v0"
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
