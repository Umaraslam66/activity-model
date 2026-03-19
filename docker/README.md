# Docker Workflow

## Purpose
This Docker image is the source build artifact for the Qwen synthetic-data generator runtime.

It is used for:
- CI builds via GitHub Actions (pushed to GHCR)
- local container smoke validation (if you have Docker)
- conversion into a Singularity `.sif` for CINECA / Leonardo

The image is intentionally **deps-only**.
Model weights are not baked into the image.

## CI Build (primary path)

The image is built automatically by GitHub Actions on push to `activity-chain-phase-a` or `main`.

See `.github/workflows/build-container.yml`.

The built image lands at:
```
ghcr.io/umaraslam66/bonzai-qwen-generator:<tag>
```

To trigger manually:
```bash
gh workflow run build-container.yml --ref activity-chain-phase-a
```

## Local Build (optional)

If you have enough disk and want to test locally:

```bash
docker build --platform linux/amd64 -f docker/Dockerfile.qwen-generator -t bonzai/qwen-generator:local .
```

## Local CPU Smoke Test
```bash
docker run --rm \
  -e OUTPUT_ROOT=/workspace/data/container-smoke \
  -v "$PWD:/workspace" \
  bonzai/qwen-generator:local \
  scripts/run_activity_chain_smoke_test.py --output-dir /workspace/data/container-smoke
```

This smoke path uses the `mock` backend and does not require a local GPU.

## Base Image

Uses `nvidia/cuda:12.4.1-runtime-ubuntu22.04` (runtime, not devel).

Why runtime instead of devel:
- vllm ships prebuilt wheels, no CUDA compilation needed at install time
- runtime image is ~2GB smaller, fits CI disk limits
- Leonardo A100 drivers are compatible with CUDA 12.4
