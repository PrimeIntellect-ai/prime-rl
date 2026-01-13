from prime_rl.utils.hf_hub import (
    build_training_path_in_repo,
    build_weights_path_in_repo,
    should_upload_step,
)


def test_should_upload_step_uses_keep_interval_only():
    assert should_upload_step(0, None) is False
    assert should_upload_step(0, 100) is True
    assert should_upload_step(1, 100) is False
    assert should_upload_step(100, 100) is True
    assert should_upload_step(200, 100) is True


def test_build_paths_default_prefixes():
    assert build_training_path_in_repo("", 123) == "checkpoints/step_123"
    assert build_weights_path_in_repo("", 123) == "weights/step_123"


def test_build_paths_custom_prefixes_are_normalized():
    assert build_training_path_in_repo("/foo/bar/", 1) == "foo/bar/checkpoints/step_1"
    assert build_weights_path_in_repo("///baz///", 2) == "baz/weights/step_2"

