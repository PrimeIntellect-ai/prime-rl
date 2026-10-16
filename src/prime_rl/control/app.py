from fastapi import FastAPI, HTTPException, status

from prime_rl.control.controller import Controller
from prime_rl.control.types import (
    Action,
    ActionNotFoundError,
    ActionRequest,
    ControlConflictError,
    InvalidActionError,
)


def create_app(controller: Controller) -> FastAPI:
    app = FastAPI(title="prime-rl control")

    @app.get("/v1/actions/{action_id}")
    async def get_action(action_id: str) -> Action:
        try:
            return controller.get(action_id)
        except ActionNotFoundError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error

    @app.post("/v1/actions", status_code=status.HTTP_202_ACCEPTED)
    async def submit_action(request: ActionRequest) -> Action:
        try:
            return controller.submit(request.type, request.step)
        except ControlConflictError as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
        except InvalidActionError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error

    return app
