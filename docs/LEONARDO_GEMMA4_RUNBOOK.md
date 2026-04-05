# Leonardo Gemma 4 Runbook

Validated against repo state and official docs on `2026-04-05`.

This project is not a public inference API deployment. It is an offline synthetic-data generator for Bonzai's activity-based mobility model. The target workload is:

- model: `google/gemma-4-31B-it`
- runtime: `vLLM`
- cluster: CINECA Leonardo
- GPUs: `2x A100 64 GB`
- output: validated activity-chain JSONL shards

## Official References

- vLLM Gemma 4 recipe:
  - <https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html>
- vLLM GPU install docs:
  - <https://docs.vllm.ai/en/latest/getting_started/installation/gpu/>
- Google Gemma Hugging Face docs:
  - <https://ai.google.dev/gemma/docs/core/huggingface_inference>
- CINECA Singularity / Apptainer docs:
  - <https://docs.hpc.cineca.it/services/singularity.html>
- CINECA access docs:
  - <https://docs.hpc.cineca.it/general/access.html>

## Deployment Choice

Use this deployment pattern:

1. Build the `linux/amd64` container in GitHub Actions.
2. Push it to GHCR.
3. Pull the exact tag on your Mac.
4. `docker save` to a tar archive.
5. Upload the tar archive to Leonardo.
6. Convert the tar archive to a Leonardo sandbox with `singularity build --sandbox`.
7. Stage Gemma weights on Leonardo storage.
8. Run the probe job.
9. Run a debug generation shard.
10. Inspect rejects before scaling out.

Do not use Hugging Face model IDs on Leonardo compute nodes. Use the staged local model path only.

Do not treat `.sif` creation as the default path for this repo. The standardized path here is Docker tar to Leonardo sandbox.

## Runtime Settings That Matter

These are the current intended Gemma 4 Leonardo settings:

- `tensor_parallel_size=2`
- `--gres=gpu:2`
- `trust_remote_code=True`
- `enforce_eager=True`
- `VLLM_USE_STANDALONE_COMPILE=0`
- `limit_mm_per_prompt={image:0,audio:0}` for text-only generation
- `HF_HUB_OFFLINE=1` on compute nodes
- `TRANSFORMERS_OFFLINE=1` on compute nodes

Why:

- vLLM's Gemma 4 recipe explicitly shows `google/gemma-4-31B-it` with TP2 for `2x A100/H100`.
- The same recipe recommends disabling multimodal allocation for text-only runs.
- CINECA docs confirm Leonardo uses A100 64 GB GPUs, CUDA 12.2, and container GPU runs should use `--nv`.
- Your repo's Leonardo scripts already encode the stable path as eager mode plus offline caches.

## CI/CD Shape

The current CI/CD path should be:

- pull requests:
  - build the container
  - run the smoke test inside the container
  - do not push
- branch pushes / main:
  - build the container
  - run the smoke test inside the container
  - push to GHCR

The canonical workflow file is:

- [.github/workflows/build-container.yml](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/.github/workflows/build-container.yml)

## Step 1: Push a Branch and Wait for CI

From the SAT repo root:

```bash
git push origin <your-branch>
```

In GitHub Actions, wait for the container workflow to finish successfully. Record the exact image tag. The workflow publishes SHA-based tags, which are the safest choice for Leonardo.

## Step 2: Pull and Save the Exact Image on Your Mac

From your Mac:

```bash
export IMAGE_TAG=<exact-ghcr-tag>
docker pull --platform linux/amd64 ghcr.io/umaraslam66/bonzai-qwen-generator:${IMAGE_TAG}
docker save ghcr.io/umaraslam66/bonzai-qwen-generator:${IMAGE_TAG} > /tmp/bonzai-qwen-generator.tar
```

Optional verification:

```bash
docker image inspect ghcr.io/umaraslam66/bonzai-qwen-generator:${IMAGE_TAG} --format '{{.Id}}'
ls -lh /tmp/bonzai-qwen-generator.tar
```

## Step 3: Upload Code and Container Tar to Leonardo

From your Mac:

```bash
export LEONARDO_USER=uaslam00
export LEONARDO_HOST=${LEONARDO_USER}@login.leonardo.cineca.it
export WORK=/leonardo_work/AIFAC_P02_222
export REMOTE_REPO=${WORK}/bonzai/sentiment-action-transformer

rsync -avz \
  --exclude .git \
  --exclude .venv \
  --exclude .venv-codex \
  --exclude .venv-smoke \
  --exclude __pycache__ \
  --exclude .pytest_cache \
  --exclude data \
  --exclude '*.sif' \
  ./ ${LEONARDO_HOST}:${REMOTE_REPO}/

scp /tmp/bonzai-qwen-generator.tar ${LEONARDO_HOST}:${WORK}/containers/
```

## Step 4: Build the Leonardo Sandbox

On Leonardo login node:

```bash
export WORK=/leonardo_work/AIFAC_P02_222
export SINGULARITY_TMPDIR=${WORK}/tmp
mkdir -p ${WORK}/containers
mkdir -p ${SINGULARITY_TMPDIR}
rm -rf ${WORK}/containers/bonzai-qwen-generator
singularity build --sandbox \
  ${WORK}/containers/bonzai-qwen-generator \
  docker-archive://${WORK}/containers/bonzai-qwen-generator.tar
```

Quick container checks:

```bash
singularity exec ${WORK}/containers/bonzai-qwen-generator python3 --version
singularity exec ${WORK}/containers/bonzai-qwen-generator cat /etc/bonzai-build-manifest.txt
singularity exec ${WORK}/containers/bonzai-qwen-generator python3 -c "import torch, transformers, vllm; print(torch.__version__); print(transformers.__version__); print(vllm.__version__)"
```

## Step 5: Stage Gemma 4 Weights on Leonardo

Run this on a Leonardo login node with internet access:

```bash
export WORK=/leonardo_work/AIFAC_P02_222
export FAST=/leonardo_scratch/fast/AIFAC_P02_222
export MODEL_DIR=${FAST}/bonzai_cache/huggingface/google--gemma-4-31B-it

mkdir -p ${FAST}/bonzai_cache/huggingface
cd ${WORK}/bonzai/sentiment-action-transformer
module load python/3.11.7
python3 -m venv .venv-smoke
source .venv-smoke/bin/activate
pip install --upgrade pip huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download google/gemma-4-31B-it --local-dir ${MODEL_DIR}
deactivate
```

Sanity check:

```bash
test -f ${MODEL_DIR}/config.json && echo model_ok
du -sh ${MODEL_DIR}
```

## Step 6: Submit the Leonardo Probe Job

On Leonardo:

```bash
export WORK=/leonardo_work/AIFAC_P02_222
export FAST=/leonardo_scratch/fast/AIFAC_P02_222
export PROJECT_ROOT=${WORK}/bonzai/sentiment-action-transformer
export CONTAINER_IMAGE=${WORK}/containers/bonzai-qwen-generator
export ABM_LLM_MODEL=${FAST}/bonzai_cache/huggingface/google--gemma-4-31B-it

cd ${PROJECT_ROOT}
sbatch scripts/cineca/activity_chain_probe_leonardo.sbatch
```

Expected probe properties:

- 2 GPUs visible
- local staged model path is used
- `trust_remote_code` is enabled
- text-only multimodal allocation is disabled
- a short generation returns a valid text response

Monitor:

```bash
squeue -u $USER
ls -t bonzai-gemma-probe-*.out | head -1
ls -t bonzai-gemma-probe-*.err | head -1
```

Inspect:

```bash
LATEST_OUT=$(ls -t bonzai-gemma-probe-*.out | head -1)
LATEST_ERR=$(ls -t bonzai-gemma-probe-*.err | head -1)
tail -100 "$LATEST_OUT"
tail -100 "$LATEST_ERR"
```

## Step 7: Submit a Debug Generation Run

Start small. The goal is not throughput yet; it is accepted records.

```bash
export WORK=/leonardo_work/AIFAC_P02_222
export FAST=/leonardo_scratch/fast/AIFAC_P02_222
export PROJECT_ROOT=${WORK}/bonzai/sentiment-action-transformer
export CONTAINER_IMAGE=${WORK}/containers/bonzai-qwen-generator
export ABM_LLM_MODEL=${FAST}/bonzai_cache/huggingface/google--gemma-4-31B-it
export OUTPUT_ROOT=${WORK}/bonzai/activity_chain_data

mkdir -p ${OUTPUT_ROOT}
cd ${PROJECT_ROOT}

sbatch \
  --export=ALL,PROJECT_ROOT=${PROJECT_ROOT},CONTAINER_IMAGE=${CONTAINER_IMAGE},ABM_LLM_MODEL=${ABM_LLM_MODEL},OUTPUT_ROOT=${OUTPUT_ROOT},TOTAL_RECORDS=100,NUM_SHARDS=1,SHARD_ID=0,BATCH_SIZE=4,MAX_TOKENS=2200,MAX_MODEL_LEN=8192,TENSOR_PARALLEL_SIZE=2 \
  scripts/cineca/activity_chain_generate_leonardo.sbatch
```

Monitor:

```bash
squeue -u $USER
ls -t bonzai-gemma-gen-*.out | head -1
ls -t bonzai-gemma-gen-*.err | head -1
```

Inspect:

```bash
LATEST_OUT=$(ls -t bonzai-gemma-gen-*.out | head -1)
LATEST_ERR=$(ls -t bonzai-gemma-gen-*.err | head -1)
tail -100 "$LATEST_OUT"
tail -100 "$LATEST_ERR"
```

## Step 8: Inspect Accepted vs Rejected Records

Expected output naming:

```bash
export RUN_NAME=activity_chain_v1
export OUTPUT_ROOT=/leonardo_work/AIFAC_P02_222/bonzai/activity_chain_data

ls -lh ${OUTPUT_ROOT}/${RUN_NAME}.shard0of1.jsonl
ls -lh ${OUTPUT_ROOT}/${RUN_NAME}.shard0of1.rejects.jsonl
```

Quick counts:

```bash
wc -l ${OUTPUT_ROOT}/${RUN_NAME}.shard0of1.jsonl
wc -l ${OUTPUT_ROOT}/${RUN_NAME}.shard0of1.rejects.jsonl
```

Sample rejects:

```bash
head -5 ${OUTPUT_ROOT}/${RUN_NAME}.shard0of1.rejects.jsonl
```

If the run shows `completed=0 rejected=N`, that is currently a generation-quality problem, not a container-hosting problem.

## What Good Looks Like

Infrastructure success means:

- GitHub Actions container build passes
- container smoke test passes in CI
- GHCR image pulls on Mac
- tar upload succeeds
- sandbox build succeeds on Leonardo
- probe job succeeds on `2x A100`
- debug generation produces at least some accepted records

Project success is stricter:

- accepted records are materially non-zero
- reject rate is low enough to justify scaling out
- output is behaviorally plausible under validation

## Known Failure Modes

### CI build fails

Check:

- workflow logs for the container build step
- whether the failure is in CUDA base image pull, PyTorch install, vLLM build, or the smoke test
- whether a dependency drifted upstream

First local reproduction:

```bash
docker build --platform linux/amd64 -f docker/Dockerfile.qwen-generator -t bonzai/qwen-generator:local .
docker run --rm bonzai/qwen-generator:local scripts/run_activity_chain_smoke_test.py --output-dir /tmp/container-smoke
```

### Probe fails on Leonardo

Check:

- `nvidia-smi -L` in the probe logs
- local model path exists
- `HF_HUB_OFFLINE=1` does not hide a missing model download
- `trust_remote_code` is still enabled
- `TENSOR_PARALLEL_SIZE=2`
- `VLLM_USE_STANDALONE_COMPILE=0`

### Generation runs but all records are rejected

That means infrastructure is mostly working. The current likely issue is schema and generation coupling:

- activity indices do not align
- leg timing does not align with activities
- summary totals do not match activities and legs

At that point the right next step is usually to reduce what the model must invent and reconstruct more fields deterministically in code.

## Files That Control the Leonardo Path

- [docker/Dockerfile.qwen-generator](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/docker/Dockerfile.qwen-generator)
- [.github/workflows/build-container.yml](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/.github/workflows/build-container.yml)
- [scripts/cineca/activity_chain_probe_leonardo.sbatch](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/scripts/cineca/activity_chain_probe_leonardo.sbatch)
- [scripts/cineca/activity_chain_generate_leonardo.sbatch](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/scripts/cineca/activity_chain_generate_leonardo.sbatch)
- [scripts/run_leonardo_gpu_probe.py](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/scripts/run_leonardo_gpu_probe.py)
- [scripts/generate_activity_chains_qwen.py](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/scripts/generate_activity_chains_qwen.py)
- [activity_chain/backends.py](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/activity_chain/backends.py)
- [activity_chain/schema.py](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/activity_chain/schema.py)
- [activity_chain/validation.py](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/activity_chain/validation.py)
