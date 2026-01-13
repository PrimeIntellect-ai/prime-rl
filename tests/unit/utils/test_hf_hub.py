from prime_rl.utils.hf_hub import HubUploadPaths, build_training_path_in_repo, build_weights_path_in_repo, should_upload_step


def test_should_upload_step_uses_keep_interval_only():
    assert should_upload_step(0, None) is False
    assert should_upload_step(0, 100) is True
    assert should_upload_step(1, 100) is False
    assert should_upload_step(100, 100) is True
    assert should_upload_step(200, 100) is True


def test_build_paths_default_prefixes():
    paths = HubUploadPaths()
    assert build_training_path_in_repo(paths, 123) == "checkpoints/step_123"
    assert build_weights_path_in_repo(paths, 123) == "weights/step_123"


def test_build_paths_custom_prefixes_are_normalized():
    paths = HubUploadPaths(training_prefix="/foo/bar/", weights_prefix="///baz///")
    assert build_training_path_in_repo(paths, 1) == "foo/bar/step_1"
    assert build_weights_path_in_repo(paths, 2) == "baz/step_2"

