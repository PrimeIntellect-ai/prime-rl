from pathlib import Path

from fastapi import FastAPI, HTTPException, status

from prime_rl.control.controller import Controller
from prime_rl.control.types import (
    Action,
    ActionNotFoundError,
    ActionRequest,
    CheckpointList,
    CheckpointStatus,
    ConfigSnapshot,
    ControlConflictError,
    InvalidActionError,
    InvalidConfigError,
    StepActionRequest,
)
from prime_rl.utils.pathing import get_all_ckpt_steps, get_step_path


def create_app(controller: Controller, ckpt_dir: Path | None = None) -> FastAPI:
    """``ckpt_dir`` is the trainer's checkpoint dir; without one, ``GET /v1/checkpoints`` answers 404."""
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
            if isinstance(request, StepActionRequest):
                return controller.submit(request.type, request.step)
            return controller.submit_config(request)
        except ControlConflictError as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
        except InvalidConfigError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, error.detail) from error
        except InvalidActionError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error

    @app.get("/v1/config")
    async def get_config() -> ConfigSnapshot:
        try:
            return controller.get_config()
        except InvalidActionError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error

    @app.get("/v1/checkpoints")
    async def get_checkpoints() -> CheckpointList:
        if ckpt_dir is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "This component has no checkpoints")
        # A step that keep_last deletes mid-scan is reported incomplete rather than failing the request.
        return CheckpointList(
            checkpoints=[
                CheckpointStatus(step=step, complete=(get_step_path(ckpt_dir, step) / "trainer" / ".metadata").exists())
                for step in get_all_ckpt_steps(ckpt_dir)
            ]
        )

    return app
