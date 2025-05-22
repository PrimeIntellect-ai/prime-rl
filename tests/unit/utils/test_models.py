from typing import get_args

import pytest

from zeroband.utils.models import ModelName


@pytest.mark.parametrize("model_name", get_args(ModelName))
def test_model_exists(hf_api, model_name):
    try:
        hf_api.model_info(model_name)
    except Exception as e:
        pytest.fail(f"Model {model_name} is not a valid Hugging Face repository: {str(e)}")
