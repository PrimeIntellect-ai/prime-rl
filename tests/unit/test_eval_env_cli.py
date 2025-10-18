import json

import pytest

from prime_rl.utils.pydantic_config import (
    extract_and_process_eval_env_args,
    merge_eval_env_into_cli_args,
    parse_eval_env_spec,
)


class TestParseEvalEnvSpec:
    def test_basic_spec(self):
        env_id, config = parse_eval_env_spec("id=gsm8k,num_examples=100")
        assert env_id == "gsm8k"
        assert config == {"num_examples": 100}

    def test_shorthand_env_id(self):
        env_id, config = parse_eval_env_spec("gsm8k,num_examples=100,temperature=0.7")
        assert env_id == "gsm8k"
        assert config == {"num_examples": 100, "temperature": 0.7}

    def test_multiple_parameters(self):
        env_id, config = parse_eval_env_spec(
            "id=math500,num_examples=50,rollouts_per_example=5,temperature=0.0,max_tokens=512"
        )
        assert env_id == "math500"
        assert config == {
            "num_examples": 50,
            "rollouts_per_example": 5,
            "temperature": 0.0,
            "max_tokens": 512,
        }

    def test_model_override(self):
        env_id, config = parse_eval_env_spec("gsm8k,model=meta-llama/Llama-3.1-70b,num_examples=100")
        assert env_id == "gsm8k"
        assert config == {"model": "meta-llama/Llama-3.1-70b", "num_examples": 100}

    def test_env_specific_args(self):
        env_id, config = parse_eval_env_spec("wordle,max_guesses=6,word_length=5")
        assert env_id == "wordle"
        assert config == {"max_guesses": 6, "word_length": 5}

    def test_json_value(self):
        env_id, config = parse_eval_env_spec("gsm8k,enable_cot=true")
        assert env_id == "gsm8k"
        assert config == {"enable_cot": True}

    def test_sampling_parameters(self):
        env_id, config = parse_eval_env_spec(
            "env,temperature=0.8,top_p=0.9,top_k=50,min_p=0.1,repetition_penalty=1.2,seed=42"
        )
        assert config == {
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
            "min_p": 0.1,
            "repetition_penalty": 1.2,
            "seed": 42,
        }

    def test_missing_env_id_raises_error(self):
        with pytest.raises(ValueError, match="must include"):
            parse_eval_env_spec("num_examples=100,temperature=0.7")

    def test_invalid_integer_raises_error(self):
        with pytest.raises(ValueError, match="Invalid integer value"):
            parse_eval_env_spec("gsm8k,num_examples=abc")

    def test_invalid_float_raises_error(self):
        with pytest.raises(ValueError, match="Invalid numeric value"):
            parse_eval_env_spec("gsm8k,temperature=xyz")


class TestExtractAndProcessEvalEnvArgs:
    def test_single_eval_env(self):
        cli_args = ["--eval.env", "gsm8k,num_examples=100", "--model.name", "gpt-4"]
        filtered_args, eval_env_dict = extract_and_process_eval_env_args(cli_args)

        assert filtered_args == ["--model.name", "gpt-4"]
        assert eval_env_dict == {"gsm8k": {"num_examples": 100}}

    def test_multiple_eval_envs(self):
        cli_args = [
            "--eval.env",
            "gsm8k,num_examples=100",
            "--eval.env",
            "math500,num_examples=50,temperature=0.0",
            "--model.name",
            "gpt-4",
        ]
        filtered_args, eval_env_dict = extract_and_process_eval_env_args(cli_args)

        assert filtered_args == ["--model.name", "gpt-4"]
        assert eval_env_dict == {
            "gsm8k": {"num_examples": 100},
            "math500": {"num_examples": 50, "temperature": 0.0},
        }

    def test_no_eval_env_args(self):
        cli_args = ["--model.name", "gpt-4", "--num-examples", "10"]
        filtered_args, eval_env_dict = extract_and_process_eval_env_args(cli_args)

        assert filtered_args == cli_args
        assert eval_env_dict == {}

    def test_missing_value_raises_error(self):
        cli_args = ["--eval.env"]
        with pytest.raises(ValueError, match="requires a value"):
            extract_and_process_eval_env_args(cli_args)


class TestMergeEvalEnvIntoCliArgs:
    def test_single_env_single_param(self):
        cli_args = ["--model.name", "gpt-4"]
        eval_env_dict = {"gsm8k": {"num_examples": 100}}

        result = merge_eval_env_into_cli_args(cli_args, eval_env_dict)

        assert "--env" in result
        env_json_index = result.index("--env") + 1
        env_json = json.loads(result[env_json_index])
        assert env_json == {"gsm8k": {"num_examples": 100}}

    def test_multiple_envs_multiple_params(self):
        cli_args = []
        eval_env_dict = {
            "gsm8k": {"num_examples": 100, "temperature": 0.0},
            "math500": {"rollouts_per_example": 5},
        }

        result = merge_eval_env_into_cli_args(cli_args, eval_env_dict)

        assert "--env" in result
        env_json_index = result.index("--env") + 1
        env_json = json.loads(result[env_json_index])
        assert env_json == eval_env_dict

    def test_per_env_mapping(self):
        cli_args = []
        eval_env_dict = {"wordle": {"rollouts_per_example": 5, "max_concurrent": 10}}

        result = merge_eval_env_into_cli_args(cli_args, eval_env_dict)

        assert "--env" in result
        env_json_index = result.index("--env") + 1
        env_json = json.loads(result[env_json_index])
        assert env_json == eval_env_dict

    def test_empty_dict(self):
        cli_args = ["--model.name", "gpt-4"]
        eval_env_dict = {}

        result = merge_eval_env_into_cli_args(cli_args, eval_env_dict)

        assert result == cli_args


class TestIntegration:
    def test_full_pipeline(self):
        cli_args = [
            "--eval.env",
            "gsm8k,num_examples=100,temperature=0.0",
            "--eval.env",
            "math500,rollouts_per_example=5",
            "--model.name",
            "gpt-4",
        ]

        filtered_args, eval_env_dict = extract_and_process_eval_env_args(cli_args)
        result = merge_eval_env_into_cli_args(filtered_args, eval_env_dict)

        assert "--model.name" in result
        assert "gpt-4" in result
        assert "--env" in result

        env_json_index = result.index("--env") + 1
        env_json = json.loads(result[env_json_index])
        assert "gsm8k" in env_json
        assert "math500" in env_json
        assert env_json["gsm8k"]["num_examples"] == 100
        assert env_json["gsm8k"]["temperature"] == 0.0
        assert env_json["math500"]["rollouts_per_example"] == 5
