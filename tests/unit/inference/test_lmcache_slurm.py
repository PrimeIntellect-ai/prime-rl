import subprocess

import pytest

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.entrypoints.inference import write_slurm_script as write_inference_slurm
from prime_rl.entrypoints.rl import write_slurm_script as write_rl_slurm


@pytest.mark.parametrize("entrypoint", ["inference", "rl"])
@pytest.mark.parametrize("deployment", ["single_node", "multi_node", "disaggregated"])
@pytest.mark.parametrize("offload", [None, "lmcache"])
def test_lmcache_slurm_rendering(tmp_path, entrypoint, deployment, offload):
    inference = {}
    if offload:
        inference["kv_cache_offload"] = {
            "type": offload,
            "cpu": {"num_bytes": 3 * 1024**3},
            "port": 9123,
            "http_port": 9124,
            "chunk_size": 512,
        }
    script_path = tmp_path / "run.sbatch"
    if entrypoint == "inference":
        config = InferenceConfig.model_validate(
            {**inference, "deployment": {"type": deployment}, "slurm": {}, "output_dir": tmp_path}
        )
        write_inference_slurm(config, tmp_path / "inference.json", tmp_path / "logs", script_path)
    else:
        config = RLConfig.model_validate(
            {
                "model": {"name": "Qwen/Qwen3-0.6B"},
                "trainer": {},
                "orchestrator": {"renderer": {"name": "default"}},
                "inference": {
                    **inference,
                    **({"deployment": {"type": deployment}} if deployment != "single_node" else {}),
                },
                "deployment": (
                    {"type": "single_node"}
                    if deployment == "single_node"
                    else {"type": "multi_node", "num_train_nodes": 1}
                ),
                "slurm": {},
                "output_dir": tmp_path,
                "run": {"name": "lmcache-test"},
            }
        )
        write_rl_slurm(config, tmp_path / "configs", tmp_path / "logs", script_path)

    script = script_path.read_text()
    subprocess.run(["bash", "-n", str(script_path)], check=True, capture_output=True)
    if "<<'LAUNCH_SH'" in script:
        body = script.split("<<'LAUNCH_SH'\n", 1)[1].rsplit("\nLAUNCH_SH", 1)[0]
        subprocess.run(["bash", "-n"], input=body, text=True, check=True, capture_output=True)
    assert script.count("lmcache server") == (1 if offload else 0)
    if offload:
        assert "--host 127.0.0.1 --port 9123" in script
        assert "--http-host 127.0.0.1 --http-port 9124" in script
        assert "--l1-size-gb 3.0" in script
        assert "--chunk-size 512" in script
        assert "http://127.0.0.1:9124/healthcheck" in script
        assert "LMCACHE_PID=$!" in script
        assert "wait -n -p FINISHED_PID" in script
