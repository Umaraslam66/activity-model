# Bonzai Activity-Chain Generator

This file is the main operator guide for the SAT directory.

## Purpose

Bonzai is an urban mobility simulator. This repository is the model-side workspace used to generate synthetic full-day activity-chain records.

Current goal:
- run `Qwen3.5-9B` on CINECA Leonardo A100 GPUs
- generate activity-chain JSONL data
- later use that synthetic dataset to train a smaller student model

The C++/CUDA simulator itself lives in a separate repository.

## Current Status

As of `2026-03-25`:

- local CPU smoke tests work
- the CI container build path works
- Docker tar -> Leonardo sandbox conversion works
- Leonardo GPU startup works
- Qwen3.5 loads successfully with `vLLM 0.18.0`
- Leonardo requires `enforce_eager=True` and `VLLM_USE_STANDALONE_COMPILE=0`
- the current blocker is no longer infrastructure
- the current blocker is output quality / validation: the model is generating JSON, but records are being rejected by strict chronology / consistency checks

In short:
- deployment is mostly solved
- schema / generation design still needs work

## Repo Layout

```text
activity_chain/
  backends.py
  prompting.py
  runner.py
  schema.py
  seeds.py
  validation.py

scripts/
  generate_activity_chains_qwen.py
  run_activity_chain_smoke_test.py
  run_leonardo_gpu_probe.py
  cineca/
    activity_chain_generate_leonardo.sbatch
    activity_chain_probe_leonardo.sbatch

docker/
  Dockerfile.qwen-generator
  README.md

spec/
  ACTIVITY_CHAIN_V1.md
  fixtures/
    activity_chain_mock_expected_first_record.json

.github/workflows/
  build-container.yml
```

## Important Runtime Facts

### Model / runtime

- model: `Qwen/Qwen3.5-9B`
- inference engine: `vLLM 0.18.0`
- transformers in container: `5.3.0`
- torch in container: `2.10.0+cu128`
- text-only mode: `language_model_only=True`

### Leonardo

- login host: `login.leonardo.cineca.it`
- partition: `boost_usr_prod`
- qos: `normal`
- GPUs: A100 64GB
- compute nodes have no internet
- login nodes can prepare containers and caches, but `.sif` builds are unreliable due to memory pressure
- use sandbox containers, not `.sif`

### Container strategy

The working path is:

```text
GitHub Actions -> GHCR image -> Mac docker pull/save -> upload tar -> Leonardo singularity build --sandbox
```

## Local Development

### Smoke test

From the SAT root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_smoke.txt
python scripts/run_activity_chain_smoke_test.py
```

This uses the mock backend only. No GPU and no model weights are required.

## Container Build

The image is built by GitHub Actions from:

- [`docker/Dockerfile.qwen-generator`](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/docker/Dockerfile.qwen-generator)
- [`requirements_inference.txt`](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/requirements_inference.txt)

Push the branch:

```bash
git push origin activity-chain-phase-a
```

Then pull the exact image tag on Mac and upload it to Leonardo as a Docker tar.

## Leonardo Manual Workflow

### 1. SSH from Mac

```bash
export LEONARDO_USER=uaslam00
export LEONARDO_HOST=${LEONARDO_USER}@login.leonardo.cineca.it
eval "$(ssh-agent -s)"
step ssh login aslamumar16@gmail.com --provisioner cineca-hpc
ssh ${LEONARDO_HOST}
```

### 2. Upload repo code from Mac

Run this from the SAT root on Mac:

```bash
rsync -avz \
  --exclude .git \
  --exclude .venv \
  --exclude .venv-codex \
  --exclude .venv-smoke \
  --exclude __pycache__ \
  --exclude .pytest_cache \
  --exclude data \
  --exclude '*.sif' \
  ./ uaslam00@login.leonardo.cineca.it:/leonardo_work/AIFAC_P02_222/bonzai/sentiment-action-transformer/
```

### 3. Upload image tar from Mac

```bash
docker pull --platform linux/amd64 ghcr.io/umaraslam66/bonzai-qwen-generator:<tag>
docker save ghcr.io/umaraslam66/bonzai-qwen-generator:<tag> > /tmp/bonzai-qwen-generator.tar
scp /tmp/bonzai-qwen-generator.tar uaslam00@login.leonardo.cineca.it:/leonardo_work/AIFAC_P02_222/containers/
```

### 4. Build Leonardo sandbox

On Leonardo:

```bash
export WORK=/leonardo_work/AIFAC_P02_222
export SINGULARITY_TMPDIR=${WORK}/tmp
mkdir -p ${SINGULARITY_TMPDIR}
rm -rf ${WORK}/containers/bonzai-qwen-generator
singularity build --sandbox \
  ${WORK}/containers/bonzai-qwen-generator \
  docker-archive://${WORK}/containers/bonzai-qwen-generator.tar
```

### 5. Verify container

On Leonardo:

```bash
export WORK=/leonardo_work/AIFAC_P02_222
singularity exec ${WORK}/containers/bonzai-qwen-generator python3 --version
singularity exec ${WORK}/containers/bonzai-qwen-generator python3 -c "import torch, transformers, vllm; print(torch.__version__); print(transformers.__version__); print(vllm.__version__)"
```

### 6. Stage model weights

On Leonardo login node:

```bash
export WORK=/leonardo_work/AIFAC_P02_222
export FAST=/leonardo_scratch/fast/AIFAC_P02_222

mkdir -p ${FAST}/bonzai_cache/huggingface
module load python/3.11.7
cd ${WORK}/bonzai/sentiment-action-transformer
python3 -m venv .venv-smoke
source .venv-smoke/bin/activate
pip install --upgrade pip huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download Qwen/Qwen3.5-9B --local-dir ${FAST}/bonzai_cache/huggingface/Qwen--Qwen3.5-9B
deactivate
```

### 7. Submit Leonardo probe

On Leonardo:

```bash
export WORK=/leonardo_work/AIFAC_P02_222
export FAST=/leonardo_scratch/fast/AIFAC_P02_222
cd ${WORK}/bonzai/sentiment-action-transformer
export PROJECT_ROOT=${WORK}/bonzai/sentiment-action-transformer
export CONTAINER_IMAGE=${WORK}/containers/bonzai-qwen-generator
export ABM_LLM_MODEL=${FAST}/bonzai_cache/huggingface/Qwen--Qwen3.5-9B
sbatch scripts/cineca/activity_chain_probe_leonardo.sbatch
```

### 8. Submit debug generation run

On Leonardo:

```bash
export WORK=/leonardo_work/AIFAC_P02_222
export FAST=/leonardo_scratch/fast/AIFAC_P02_222
cd ${WORK}/bonzai/sentiment-action-transformer
export PROJECT_ROOT=${WORK}/bonzai/sentiment-action-transformer
export ABM_LLM_MODEL=${FAST}/bonzai_cache/huggingface/Qwen--Qwen3.5-9B
export CONTAINER_IMAGE=${WORK}/containers/bonzai-qwen-generator
export OUTPUT_ROOT=${WORK}/bonzai/activity_chain_data
export TOTAL_RECORDS=100
export NUM_SHARDS=1
export SHARD_ID=0
sbatch scripts/cineca/activity_chain_generate_leonardo.sbatch
```

### 9. Monitor

On Leonardo:

```bash
squeue -u $USER
ls -t bonzai-qwen-gen-*.out | head -1
ls -t bonzai-qwen-gen-*.err | head -1
```

Inspect the newest logs:

```bash
LATEST_OUT=$(ls -t bonzai-qwen-gen-*.out | head -1)
LATEST_ERR=$(ls -t bonzai-qwen-gen-*.err | head -1)
tail -50 "$LATEST_OUT"
tail -50 "$LATEST_ERR"
```

## Current Leonardo Defaults

The current Slurm wrapper intentionally uses:

- `--cleanenv`
- `--nv`
- persistent caches on `$FAST`
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `ABM_VLLM_ENFORCE_EAGER=1`
- `VLLM_USE_STANDALONE_COMPILE=0`

Those settings reflect what has actually worked on Leonardo so far.

## What Has Been Solved

- Qwen3.5 architecture recognition works with `transformers==5.3.0`
- Leonardo sandbox builds work from Docker tar archives
- GPU startup works on Leonardo
- vLLM can load the model successfully
- eager mode works; compile mode does not currently work on Leonardo for this stack
- structured output schema issues caused by `uniqueItems` have been removed from the vLLM-facing schema

## Current Blocker

The current blocker is **record rejection**, not deployment.

The model is producing JSON, but the outputs often violate strict cross-field constraints, for example:

- activity sequences start at `1` instead of `0`
- leg sequencing does not line up with activity indices
- leg arrival times do not match the next activity start
- summary totals do not match the generated legs

Because of that, current Leonardo runs can produce:

- `completed=0`
- `rejected=N`

This means the next major work item is likely a schema/pipeline redesign:

- let the model generate fewer dependent fields
- compute chronology, legs, and summary deterministically in code
- keep validation strict after deterministic reconstruction

## Files To Keep As Source Of Truth

Use these files first:

- [`claude.md`](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/claude.md)
- [`AGENT_PROMPT.md`](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/AGENT_PROMPT.md)
- [`docker/Dockerfile.qwen-generator`](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/docker/Dockerfile.qwen-generator)
- [`scripts/cineca/activity_chain_generate_leonardo.sbatch`](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/scripts/cineca/activity_chain_generate_leonardo.sbatch)
- [`scripts/cineca/activity_chain_probe_leonardo.sbatch`](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/scripts/cineca/activity_chain_probe_leonardo.sbatch)
- [`activity_chain/schema.py`](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/activity_chain/schema.py)
- [`activity_chain/validation.py`](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/activity_chain/validation.py)

## Notes

- CINECA may say "Apptainer" while the commands still use `singularity`. On Leonardo that is normal; the runtime is compatible.
- Do not rely on `.sif` creation on login nodes.
- Do not use Hugging Face model IDs on compute nodes. Always use the local staged model path.
- Do not assume active queue jobs remain queryable with `squeue`; once they finish, use `sacct`.
