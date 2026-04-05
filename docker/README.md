# Docker Workflow

This directory defines the container build used for Leonardo runs.

## Purpose

The Docker image is the source artifact for the Leonardo runtime. It is:

- built in GitHub Actions as `linux/amd64`
- pushed to GHCR
- pulled on Mac
- saved as a Docker tar
- uploaded to Leonardo
- converted on Leonardo into a Singularity sandbox with `singularity build --sandbox`

Model weights are not baked into the image.

## Current Build

- Dockerfile: [`docker/Dockerfile.qwen-generator`](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/docker/Dockerfile.qwen-generator)
- base image: `nvidia/cuda:12.2.2-devel-ubuntu22.04`
- Python runtime: `python3`
- key runtime packages:
  - `torch==2.5.0` from the CUDA 12.1 wheel index
  - `transformers==4.57.1`
  - `accelerate==1.13.0`

Historical filenames still mention `qwen`, but the current rollout target is Gemma 4.

The image is intentionally dependency-focused. The repository code is copied into the image, but model weights and cluster caches live outside the image.

For Gemma 4 on Leonardo, the repo now uses the `Transformers + Accelerate` path instead of `vLLM`. The `vLLM` path drifted onto a newer CUDA / PyTorch stack than Leonardo can support cleanly.

## CI Build

GitHub Actions workflow:

- [build-container.yml](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/.github/workflows/build-container.yml)

Current CI behavior:

- pull requests build the container and run the mock smoke test inside the image
- branch pushes build, smoke-test, and then push the image to GHCR
- BuildKit cache is stored in GitHub Actions to avoid full rebuilds on every change

The image is published to:

```text
ghcr.io/umaraslam66/bonzai-qwen-generator:<tag>
```

## Local Build

Optional local build:

```bash
docker build --platform linux/amd64 -f docker/Dockerfile.qwen-generator -t bonzai/qwen-generator:local .
```

## Local Smoke Test

Mock backend only:

```bash
docker run --rm \
  -e OUTPUT_ROOT=/workspace/data/container-smoke \
  -v "$PWD:/workspace" \
  bonzai/qwen-generator:local \
  scripts/run_activity_chain_smoke_test.py --output-dir /workspace/data/container-smoke
```

## Leonardo Transfer Pattern

The current working transfer pattern is:

```bash
docker pull --platform linux/amd64 ghcr.io/umaraslam66/bonzai-qwen-generator:<tag>
docker save ghcr.io/umaraslam66/bonzai-qwen-generator:<tag> > /tmp/bonzai-qwen-generator.tar
scp /tmp/bonzai-qwen-generator.tar uaslam00@login.leonardo.cineca.it:/leonardo_work/AIFAC_P02_222/containers/
```

Then on Leonardo:

```bash
export WORK=/leonardo_work/AIFAC_P02_222
export SINGULARITY_TMPDIR=${WORK}/tmp
mkdir -p ${SINGULARITY_TMPDIR}
singularity build --sandbox \
  ${WORK}/containers/bonzai-qwen-generator \
  docker-archive://${WORK}/containers/bonzai-qwen-generator.tar
```

`.sif` creation is intentionally not the main workflow here because login-node compression has been unreliable for large images.

Avoid treating `singularity pull docker://...` to a `.sif` as the default Leonardo path for this project. The repo is standardized on Docker tar upload followed by `singularity build --sandbox`.
