# Inference Dashboard

Standalone side utility for monitoring disaggregated vLLM inference jobs.

## Install

```bash
cd tools/inference_dashboard
uv sync
npm install
npm run build
```

## Run

```bash
uv run inference-dashboard --host 0.0.0.0 --port 8050
```

Then open `http://localhost:8050`.
