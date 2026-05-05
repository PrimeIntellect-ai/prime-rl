from pathlib import Path

from prime_rl.baselines.config import load_config


def test_load_config(tmp_path: Path):
    cfg_path = tmp_path / "baseline.toml"
    cfg_path.write_text(
        """
env_id = "hf_singleturn"
model = "google/gemma"
output_dir = "outputs/baseline"
protocol = "maj_at_n"
num_examples = 4
rollouts_per_example = 3
score_max_concurrency = 128
ks = [1, 3]
base_url = "http://127.0.0.1:8000/v1"
api_profile = "vllm_permissive"
client_type = "openai_completions"

[env_args]
dataset_name = "joanvelja/gpqa-open-ended"
task_type = "open_ended"

[sampling]
temperature = 0.7
max_tokens = 1024

[launch]
mode = "external"
dp = 4
gpus = "0,1,2,3"
launch_prefix = "srun --ntasks=1 --gpus=4 --exclusive"
wait_timeout_s = 120.0
server_start_retries = 3
chat_template = "configs/baselines/base_text_chat_template.jinja"
vllm_extra = {skip_mm_profiling = true}
""".strip()
    )

    config = load_config(cfg_path)

    assert config.env_id == "hf_singleturn"
    assert config.rollouts_per_example == 3
    assert config.score_max_concurrency == 128
    assert config.api_profile == "vllm_permissive"
    assert config.client_type == "openai_completions"
    assert config.env_args["task_type"] == "open_ended"
    assert config.sampling_args["max_tokens"] == 1024
    assert config.launch.dp == 4
    assert config.launch.launch_prefix == ["srun", "--ntasks=1", "--gpus=4", "--exclusive"]
    assert config.launch.wait_timeout_s == 120.0
    assert config.launch.server_start_retries == 3
    assert config.launch.chat_template == "configs/baselines/base_text_chat_template.jinja"
    assert config.launch.vllm_extra == {"skip_mm_profiling": True}
