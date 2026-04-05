# Bonzai Activity-Chain Generator

This file is the main operator guide for the SAT directory.

## Purpose

Bonzai is an urban mobility simulator. This repository is the model-side workspace used to generate synthetic full-day activity-chain records.

Current goal:
- run `google/gemma-4-31B-it` on CINECA Leonardo A100 GPUs
- generate activity-chain JSONL data
- later use that synthetic dataset to train a smaller student model

The C++/CUDA simulator itself lives in a separate repository.

## Current Status

As of `2026-04-05`:

- local CPU smoke tests work
- the CI container build path is now based on `Transformers + Accelerate`
- Docker tar -> Leonardo sandbox conversion works
- Leonardo GPU startup works
- the previous Qwen3.5 path was stabilized on Leonardo
- the current rollout target is Gemma 4
- the previous `vLLM` path was dropped for Gemma 4 because released Gemma 4 support moved onto a newer CUDA / PyTorch stack than Leonardo supports cleanly
- the current blocker is no longer infrastructure
- the current blocker is output quality / validation: the model is generating JSON, but records are being rejected by strict chronology / consistency checks

In short:
- deployment is mostly solved
- schema / generation design still needs work
- Gemma 4 needs a fresh Leonardo validation pass

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

- current model target: `google/gemma-4-31B-it`
- inference engine target: `Transformers + Accelerate`
- transformers target in container: `4.57.1`
- torch target in container: `2.5.0` from the CUDA 12.1 wheel index
- Gemma 4 Leonardo runs should default to `device_map=auto`
- Gemma 4 Leonardo runs should default to `attn_implementation=sdpa`
- Gemma 4 Leonardo runs should default to `trust_remote_code=False`

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

### OpenRouter probe

Use this to test prompt and schema ideas locally before spending Leonardo GPU time.

The script is:

- [`scripts/probe_openrouter_qwen.py`](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/scripts/probe_openrouter_qwen.py)

Export your API key in the shell:

```bash
export OPENROUTER_API_KEY=...
```

Basic free-form prompt probe:

```bash
python scripts/probe_openrouter_qwen.py \
  --user-prompt "Generate a plausible weekday activity skeleton for a 32-year-old parent in a transit-rich city."
```

Probe the current Bonzai prompt using a locally generated seed:

```bash
python scripts/probe_openrouter_qwen.py \
  --seed-index 0 \
  --dump-content /tmp/qwen-probe.txt
```

Probe with the current activity-chain schema:

```bash
python scripts/probe_openrouter_qwen.py \
  --seed-index 0 \
  --use-activity-schema \
  --dump-response /tmp/qwen-probe-response.json
```

The script sends requests to the OpenRouter chat completions API and supports `response_format` with `json_schema`, which is the documented OpenRouter structured-output path.

## Gemma 4 Constraints

Historical filenames still mention `qwen`, but the active Leonardo target is now Gemma 4.

Official Gemma 4 / Transformers guidance implies:

- `google/gemma-4-31B-it` is not a 1-GPU drop-in on Leonardo
- on Leonardo, use both A100s and let Accelerate shard the model with `device_map=auto`
- use the local staged model path, not a remote model ID, on compute nodes
- keep the container path simple: PyTorch + Transformers + Accelerate, not source-built `vLLM`

## Container Build

The image is built by GitHub Actions from:

- [`docker/Dockerfile.qwen-generator`](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/docker/Dockerfile.qwen-generator)
- [`requirements_inference.txt`](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/requirements_inference.txt)

Current CI expectations:

- pull requests build the image and run the mock smoke test inside the container
- branch pushes build, smoke-test, and then push to GHCR
- GitHub Actions cache is enabled to avoid recompiling the whole image on every run

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
singularity exec ${WORK}/containers/bonzai-qwen-generator python3 -c "import torch, transformers, accelerate; print(torch.__version__); print(transformers.__version__); print(accelerate.__version__)"
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
hf download google/gemma-4-31B-it --local-dir ${FAST}/bonzai_cache/huggingface/google--gemma-4-31B-it
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
export ABM_LLM_MODEL=${FAST}/bonzai_cache/huggingface/google--gemma-4-31B-it
sbatch scripts/cineca/activity_chain_probe_leonardo.sbatch
```

### 8. Submit debug generation run

On Leonardo:

```bash
export WORK=/leonardo_work/AIFAC_P02_222
export FAST=/leonardo_scratch/fast/AIFAC_P02_222
cd ${WORK}/bonzai/sentiment-action-transformer
export PROJECT_ROOT=${WORK}/bonzai/sentiment-action-transformer
export ABM_LLM_MODEL=${FAST}/bonzai_cache/huggingface/google--gemma-4-31B-it
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
ls -t bonzai-gemma-gen-*.out | head -1
ls -t bonzai-gemma-gen-*.err | head -1
```

Inspect the newest logs:

```bash
LATEST_OUT=$(ls -t bonzai-gemma-gen-*.out | head -1)
LATEST_ERR=$(ls -t bonzai-gemma-gen-*.err | head -1)
tail -50 "$LATEST_OUT"
tail -50 "$LATEST_ERR"
```

## Current Leonardo Defaults

The current Slurm wrapper intentionally uses:

- `--cleanenv`
- `--nv`
- `--gres=gpu:2`
- persistent caches on `$FAST`
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `ABM_BACKEND=transformers`
- `ABM_DEVICE_MAP=auto`
- `ABM_ATTN_IMPLEMENTATION=sdpa`
- `ABM_TRUST_REMOTE_CODE=0`

Those settings reflect what has actually worked on Leonardo so far.

## What Has Been Solved

- the older Qwen3.5 Leonardo path was debugged end-to-end
- Leonardo sandbox builds work from Docker tar archives
- GPU startup works on Leonardo
- the Transformers + Accelerate container path works on Leonardo's CUDA 12.2 stack
- the probe and generation scripts now target the staged local Gemma 4 path directly
- CI can validate the container without rebuilding a source `vLLM` stack

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
- [`docs/LEONARDO_GEMMA4_RUNBOOK.md`](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/docs/LEONARDO_GEMMA4_RUNBOOK.md)
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
