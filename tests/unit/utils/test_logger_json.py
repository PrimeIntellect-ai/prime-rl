import json

from prime_rl.utils.logger import reset_logger, setup_logger


def test_setup_logger_json_emits_valid_json(capsys):
    # Reset the autouse logger so we can reconfigure it for this test.
    reset_logger()

    logger = setup_logger("info", json=True)
    logger.info("hello-json")

    out = capsys.readouterr().out.strip().splitlines()
    assert out, "Expected at least one log line on stdout"

    payload = json.loads(out[-1])
    assert payload["record"]["message"] == "hello-json"
