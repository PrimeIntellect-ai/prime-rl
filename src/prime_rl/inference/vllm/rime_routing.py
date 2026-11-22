"""Use vLLM's internal DP scheduler when a pod GPU is reserved for a sidecar."""


def install_internal_dp_routing(app):
    # vLLM can prebuild its middleware stack before returning the FastAPI app.
    # Wrap that stack intact, including authentication and endpoint middleware.
    app.middleware_stack = InternalDPRouting(app.middleware_stack or app.build_middleware_stack())


class InternalDPRouting:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope = {
                **scope,
                "headers": [(key, value) for key, value in scope["headers"] if key.lower() != b"x-data-parallel-rank"],
            }
        await self.app(scope, receive, send)
