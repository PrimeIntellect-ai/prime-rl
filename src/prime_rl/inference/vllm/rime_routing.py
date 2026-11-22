"""Use vLLM's internal DP scheduler when a pod GPU is reserved for a sidecar."""


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
