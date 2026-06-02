# rollout-viewer

Online rollout visualization for PrimeRL runs.

The package provides:

- `rollout-sync`: parse PrimeRL rollout dumps, join optional diagnostics sidecars, and write compact viewer artifacts.
- `rollout-serve`: serve the artifact store through a FastAPI backend and static web UI.

The producer/store/viewer contracts live in `rollout_viewer/contracts.md`.
