# Leonardo Fresh Start Manual Runbook

This runbook avoids the SSH wrapper scripts and uses manual commands only.

## Scope Of Cleanup

The cleanup commands below remove only Bonzai-specific artifacts on Leonardo:

- repo checkout at `$WORK/bonzai/sentiment-action-transformer`
- generated outputs at `$WORK/bonzai/activity_chain_data`
- container sandbox and tar at `$WORK/containers/bonzai-qwen-generator*`
- Bonzai caches at `$FAST/bonzai_cache/*`
- Bonzai Slurm logs in the repo directory

They do **not** remove unrelated project files in `$WORK` or `$FAST`.

## Fresh Start Sequence

### 1. Mac: push the current branch and wait for CI

From the local git root:

```bash
cd /Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT
git status --short
git add commands.md scripts/cineca/activity_chain_generate_leonardo.sbatch scripts/cineca/activity_chain_probe_leonardo.sbatch scripts/run_leonardo_gpu_probe.py spec/LEONARDO_ROBUST_DEPLOYMENT_PLAN.md
git commit -m "Harden Leonardo runtime and add GPU probe"
git push origin activity-chain-phase-a
```

Wait for GitHub Actions `Build and Push Generator Container` to finish green.

### 2. Mac: define shared variables

```bash
cd /Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT
export LEONARDO_USER=uaslam00
export LEONARDO_HOST=${LEONARDO_USER}@login.leonardo.cineca.it
export LEONARDO_WORK=/leonardo_work/AIFAC_P02_222
export LEONARDO_FAST=/leonardo_scratch/fast/AIFAC_P02_222
export REMOTE_CODE=${LEONARDO_WORK}/bonzai/sentiment-action-transformer
export REMOTE_CONTAINER_DIR=${LEONARDO_WORK}/containers
export REMOTE_CONTAINER_TAR=${REMOTE_CONTAINER_DIR}/bonzai-qwen-generator.tar
export REMOTE_CONTAINER_SANDBOX=${REMOTE_CONTAINER_DIR}/bonzai-qwen-generator
export IMAGE_TAG=$(git rev-parse --short HEAD)
export IMAGE=ghcr.io/umaraslam66/bonzai-qwen-generator:${IMAGE_TAG}
```

### 3. Mac: authenticate SSH

```bash
eval "$(ssh-agent -s)"
step ssh login aslamumar16@gmail.com --provisioner cineca-hpc
ssh ${LEONARDO_HOST}
exit
```

### 4. Leonardo: clean Bonzai artifacts only

SSH in first:

```bash
ssh ${LEONARDO_HOST}
tmux new -s bonzai
```

Then run:

```bash
export WORK=/leonardo_work/AIFAC_P02_222
export FAST=/leonardo_scratch/fast/AIFAC_P02_222

rm -rf ${WORK}/bonzai/sentiment-action-transformer
rm -rf ${WORK}/bonzai/activity_chain_data
rm -rf ${WORK}/containers/bonzai-qwen-generator
rm -f  ${WORK}/containers/bonzai-qwen-generator.tar
rm -rf ${FAST}/bonzai_cache/huggingface
rm -rf ${FAST}/bonzai_cache/vllm
rm -rf ${FAST}/bonzai_cache/triton
mkdir -p ${WORK}/bonzai
mkdir -p ${WORK}/containers
mkdir -p ${WORK}/tmp
mkdir -p ${WORK}/bonzai/activity_chain_data
mkdir -p ${FAST}/bonzai_cache/huggingface
mkdir -p ${FAST}/bonzai_cache/vllm
mkdir -p ${FAST}/bonzai_cache/triton
exit
```

### 5. Mac: sync the repo checkout

```bash
cd /Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT
rsync -avz \
  --exclude .git \
  --exclude .venv \
  --exclude .venv-codex \
  --exclude .venv-smoke \
  --exclude __pycache__ \
  --exclude .pytest_cache \
  --exclude data \
  --exclude '*.sif' \
  ./ ${LEONARDO_HOST}:${REMOTE_CODE}/
```

### 6. Mac: pull the image by commit SHA and stream it to Leonardo

```bash
docker system prune -af
docker volume prune -af
docker pull --platform linux/amd64 ${IMAGE}
docker save ${IMAGE} | ssh ${LEONARDO_HOST} "cat > ${REMOTE_CONTAINER_TAR}"
```

### 7. Leonardo: build a fresh sandbox from the tar

```bash
ssh ${LEONARDO_HOST}
tmux attach -t bonzai || tmux new -s bonzai
export WORK=/leonardo_work/AIFAC_P02_222
export SINGULARITY_TMPDIR=${WORK}/tmp
mkdir -p ${SINGULARITY_TMPDIR}
rm -rf ${WORK}/containers/bonzai-qwen-generator
singularity build --sandbox \
  ${WORK}/containers/bonzai-qwen-generator \
  docker-archive://${WORK}/containers/bonzai-qwen-generator.tar
```

### 8. Leonardo: verify the container before using GPUs

```bash
export WORK=/leonardo_work/AIFAC_P02_222
singularity exec ${WORK}/containers/bonzai-qwen-generator python3 --version
singularity exec ${WORK}/containers/bonzai-qwen-generator python3 -c "import torch, transformers, vllm; print(torch.__version__); print(transformers.__version__); print(vllm.__version__)"
singularity exec ${WORK}/containers/bonzai-qwen-generator gcc --version
```

### 9. Leonardo: if model weights are gone, re-download them

Run this only if `${FAST}/bonzai_cache/huggingface/Qwen--Qwen3.5-9B` is empty after cleanup.

```bash
module load python/3.11.7
cd ${WORK}/bonzai/sentiment-action-transformer
python3 -m venv .venv-smoke
source .venv-smoke/bin/activate
pip install --upgrade pip huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download Qwen/Qwen3.5-9B \
  --local-dir ${FAST}/bonzai_cache/huggingface/Qwen--Qwen3.5-9B
deactivate
```

### 10. Leonardo: submit the probe job first

```bash
export WORK=/leonardo_work/AIFAC_P02_222
export FAST=/leonardo_scratch/fast/AIFAC_P02_222
cd ${WORK}/bonzai/sentiment-action-transformer
export PROJECT_ROOT=${WORK}/bonzai/sentiment-action-transformer
export CONTAINER_IMAGE=${WORK}/containers/bonzai-qwen-generator
export ABM_LLM_MODEL=${FAST}/bonzai_cache/huggingface/Qwen--Qwen3.5-9B
sbatch scripts/cineca/activity_chain_probe_leonardo.sbatch
```

Monitor:

```bash
squeue -u $USER
tail -f bonzai-qwen-probe-*.out
tail -f bonzai-qwen-probe-*.err
```

### 11. Leonardo: only after probe success, submit the 100-record debug run

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

Monitor:

```bash
squeue -u $USER
tail -f bonzai-qwen-gen-*.out
tail -f bonzai-qwen-gen-*.err
```

### 12. Leonardo: production only after debug success

```bash
export WORK=/leonardo_work/AIFAC_P02_222
export FAST=/leonardo_scratch/fast/AIFAC_P02_222
cd ${WORK}/bonzai/sentiment-action-transformer
export PROJECT_ROOT=${WORK}/bonzai/sentiment-action-transformer
export ABM_LLM_MODEL=${FAST}/bonzai_cache/huggingface/Qwen--Qwen3.5-9B
export CONTAINER_IMAGE=${WORK}/containers/bonzai-qwen-generator
export OUTPUT_ROOT=${WORK}/bonzai/activity_chain_data
export TOTAL_RECORDS=50000
export NUM_SHARDS=8
for i in $(seq 0 7); do
  export SHARD_ID=$i
  sbatch scripts/cineca/activity_chain_generate_leonardo.sbatch
done
```
