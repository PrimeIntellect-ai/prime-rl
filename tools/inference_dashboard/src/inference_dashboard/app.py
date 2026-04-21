from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from inference_dashboard.models import FromSlurmRequest, Topology
from inference_dashboard.monitor import JobMonitor
from inference_dashboard.slurm import resolve_from_slurm


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = PACKAGE_ROOT / "frontend" / "dist"
FRONTEND_ASSETS = FRONTEND_DIST / "assets"

app = FastAPI(title="Inference Dashboard", version="0.1.0")
monitors: dict[int, JobMonitor] = {}

if FRONTEND_ASSETS.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS), name="assets")


@app.get("/")
async def index() -> FileResponse:
    index_path = FRONTEND_DIST / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=503,
            detail="frontend not built; run `npm install && npm run build` in tools/inference_dashboard",
        )
    return FileResponse(index_path)


async def ensure_monitor(request: FromSlurmRequest) -> tuple[Topology, JobMonitor]:
    topology = await resolve_from_slurm(request)
    monitor = monitors.get(request.job_id)
    if monitor is None:
        monitor = JobMonitor(topology)
        monitors[request.job_id] = monitor
        await monitor.refresh_once()
        await monitor.start()
    else:
        monitor.topology = topology
    return topology, monitor


@app.post("/api/topology/from-slurm")
async def topology_from_slurm(request: FromSlurmRequest) -> Topology:
    topology, _ = await ensure_monitor(request)
    return topology


@app.get("/api/jobs/{job_id}/topology")
async def get_topology(job_id: int) -> Topology:
    monitor = monitors.get(job_id)
    if monitor is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} is not being monitored")
    return monitor.topology


@app.get("/api/jobs/{job_id}/snapshot")
async def get_snapshot(job_id: int):
    monitor = monitors.get(job_id)
    if monitor is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} is not being monitored")
    if monitor.latest_snapshot is None:
        raise HTTPException(status_code=503, detail="snapshot not ready yet")
    return monitor.latest_snapshot


@app.get("/api/jobs/{job_id}/stream")
async def stream_snapshot(job_id: int) -> StreamingResponse:
    monitor = monitors.get(job_id)
    if monitor is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} is not being monitored")

    async def event_stream():
        last_timestamp = -1.0
        while True:
            snapshot = monitor.latest_snapshot
            if snapshot is not None and snapshot.timestamp != last_timestamp:
                last_timestamp = snapshot.timestamp
                yield f"data: {snapshot.model_dump_json()}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference metrics dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    uvicorn.run("inference_dashboard.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
