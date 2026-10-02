import pytest

grpc = pytest.importorskip("grpc")
pytest.importorskip("modelexpress_rl")

from prime_rl.transports.weights.mx_refit import version_leased  # noqa: E402


class _RpcError(grpc.RpcError):
    def __init__(self, code, details):
        self._code, self._details = code, details

    def code(self):
        return self._code

    def details(self):
        return self._details


def test_active_lease_is_recognized_as_a_live_reader():
    error = _RpcError(grpc.StatusCode.FAILED_PRECONDITION, "weight version has an active lease")
    assert version_leased(error)


@pytest.mark.parametrize(
    "code, details",
    [
        # The server uses the same status for a shard changing mid-delete.
        (grpc.StatusCode.FAILED_PRECONDITION, "weight version shard changed while it was being deleted"),
        (grpc.StatusCode.NOT_FOUND, "weight version shard not found"),
        (grpc.StatusCode.FAILED_PRECONDITION, None),
    ],
)
def test_other_release_failures_are_not_blamed_on_a_reader(code, details):
    assert not version_leased(_RpcError(code, details))
