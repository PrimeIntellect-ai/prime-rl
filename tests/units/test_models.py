import pytest
from huggingface_hub import HfApi
from typing import get_args
from zeroband.utils.models import ModelName


@pytest.fixture(scope="session")
def api():
    return HfApi()


@pytest.mark.parametrize("model_name", get_args(ModelName))
def test_model_exists(api, model_name):
    try:
        api.model_info(model_name)
    except Exception as e:
        pytest.fail(f"Model {model_name} is not a valid Hugging Face repository: {str(e)}")
