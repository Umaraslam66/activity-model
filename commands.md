# Bonzai Activity-Chain: All Commands (Start to End)

## Recovery: Clean Up After a Failed Pull/Build

**Run this FIRST if a previous docker pull or save crashed or filled disk:**

```bash
# Step 1: Quit Docker Desktop completely
osascript -e 'quit app "Docker"'

# Step 2: Delete the Docker VM disk (this is safe — reclaims all Docker space)
rm -rf ~/Library/Containers/com.docker.docker/Data/vms/0/data/Docker.raw

# Step 3: Restart Docker Desktop (it recreates a fresh VM disk)
open -a Docker
# Wait ~30 seconds for it to start

# Step 4: Verify space recovered
df -h /
# Should show ~30GB+ free

# Step 5: Remove any leftover tar files
rm -f /tmp/bonzai-qwen-generator.tar
```

**If Docker Desktop won't start after deleting the VM disk:**
```bash
rm -rf ~/Library/Containers/com.docker.docker/Data
open -a Docker
```

---

## Phase 1: SSH into Leonardo (from Mac)

```bash
eval "$(ssh-agent -s)"
step ssh login aslamumar16@gmail.com --provisioner cineca-hpc
# Complete browser auth when prompted
# Certificate valid for ~12 hours

ssh uaslam00@login.leonardo.cineca.it

# Start tmux immediately
tmux new -s bonzai
# To reattach after disconnect: tmux attach -t bonzai
```

## Phase 2: Upload Code to Leonardo (from Mac)

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
  ./ uaslam00@login.leonardo.cineca.it:/leonardo_work/AIFAC_P02_222/bonzai/sentiment-action-transformer/
```

## Phase 3: Build & Ship Container (from Mac)

### 3a. CI build
Push to `activity-chain-phase-a` branch triggers GitHub Actions.

### 3b. Clean Docker, pull, and stream to Leonardo

```bash
# Clean any leftover images first
docker system prune -af
docker volume prune -af

# Pull
docker pull --platform linux/amd64 ghcr.io/umaraslam66/bonzai-qwen-generator:activity-chain-phase-a

# Stream directly to Leonardo (no local tar file needed)
docker save ghcr.io/umaraslam66/bonzai-qwen-generator:activity-chain-phase-a | \
  ssh uaslam00@login.leonardo.cineca.it \
  'cat > /leonardo_work/AIFAC_P02_222/containers/bonzai-qwen-generator.tar'

# Clean up after successful upload
docker rmi ghcr.io/umaraslam66/bonzai-qwen-generator:activity-chain-phase-a
docker system prune -af
```

### 3c. Build sandbox on Leonardo (inside tmux)

```bash
rm -rf $WORK/containers/bonzai-qwen-generator
export SINGULARITY_TMPDIR=$WORK/tmp
mkdir -p $SINGULARITY_TMPDIR
singularity build --sandbox $WORK/containers/bonzai-qwen-generator \
  docker-archive://$WORK/containers/bonzai-qwen-generator.tar
```

### 3d. Verify container

```bash
singularity exec $WORK/containers/bonzai-qwen-generator python3 --version
singularity exec $WORK/containers/bonzai-qwen-generator python3 -c \
  "import transformers; print('transformers', transformers.__version__)"
# Expected: transformers 5.3.0
```

## Phase 4: Leonardo Environment Setup (one-time)

### Cache directories
```bash
mkdir -p $FAST/bonzai_cache/huggingface
mkdir -p $FAST/bonzai_cache/vllm
mkdir -p $FAST/bonzai_cache/triton
```

### Download model weights (login node, inside tmux)
```bash
module load python/3.11.7
cd $WORK/bonzai/sentiment-action-transformer
python3 -m venv .venv-smoke
source .venv-smoke/bin/activate
pip install --upgrade pip huggingface_hub hf_transfer

export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download Qwen/Qwen3.5-9B \
  --local-dir $FAST/bonzai_cache/huggingface/Qwen--Qwen3.5-9B
```

## Phase 5: Submit Probe Job (imports + Triton + tiny vLLM run)

```bash
cd $WORK/bonzai/sentiment-action-transformer

export PROJECT_ROOT=$WORK/bonzai/sentiment-action-transformer
export ABM_LLM_MODEL=$FAST/bonzai_cache/huggingface/Qwen--Qwen3.5-9B
export CONTAINER_IMAGE=$WORK/containers/bonzai-qwen-generator

sbatch scripts/cineca/activity_chain_probe_leonardo.sbatch
```

### Monitor
```bash
squeue -u $USER
tail -f bonzai-qwen-probe-<JOBID>.out
tail -f bonzai-qwen-probe-<JOBID>.err
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed
```

Only move on after the probe passes.

## Phase 6: Submit Debug Job (100 records, 1 GPU)

```bash
cd $WORK/bonzai/sentiment-action-transformer

export PROJECT_ROOT=$WORK/bonzai/sentiment-action-transformer
export ABM_LLM_MODEL=$FAST/bonzai_cache/huggingface/Qwen--Qwen3.5-9B
export CONTAINER_IMAGE=$WORK/containers/bonzai-qwen-generator
export OUTPUT_ROOT=$WORK/bonzai/activity_chain_data
export TOTAL_RECORDS=100
export NUM_SHARDS=1
export SHARD_ID=0

sbatch scripts/cineca/activity_chain_generate_leonardo.sbatch
```

### Monitor
```bash
squeue -u $USER
tail -f bonzai-qwen-gen-<JOBID>.out
tail -f bonzai-qwen-gen-<JOBID>.err
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed
```

### Check outputs
```bash
wc -l $WORK/bonzai/activity_chain_data/activity_chain_v1.shard0of1.jsonl
head -1 $WORK/bonzai/activity_chain_data/activity_chain_v1.shard0of1.jsonl | python3 -m json.tool
```

## Phase 7: Production Sharded Run (50k records)

```bash
cd $WORK/bonzai/sentiment-action-transformer

export PROJECT_ROOT=$WORK/bonzai/sentiment-action-transformer
export ABM_LLM_MODEL=$FAST/bonzai_cache/huggingface/Qwen--Qwen3.5-9B
export CONTAINER_IMAGE=$WORK/containers/bonzai-qwen-generator
export OUTPUT_ROOT=$WORK/bonzai/activity_chain_data
export TOTAL_RECORDS=50000
export NUM_SHARDS=8

for i in $(seq 0 7); do
  export SHARD_ID=$i
  sbatch scripts/cineca/activity_chain_generate_leonardo.sbatch
done
```

## Cleanup

### Mac: Free Docker disk space
```bash
docker rmi ghcr.io/umaraslam66/bonzai-qwen-generator:activity-chain-phase-a 2>/dev/null
docker system prune -af
docker volume prune -af
rm -f /tmp/bonzai-qwen-generator.tar
# If still low on space: quit Docker Desktop → delete VM disk → restart (see Recovery section)
```

### Leonardo: Remove old artifacts
```bash
rm -rf $WORK/containers/bonzai-qwen-generator
rm -f  $WORK/containers/bonzai-qwen-generator.tar
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Permission denied (publickey)` | Re-run `step ssh login` |
| SSH drops mid-operation | Use `tmux`; reattach with `tmux attach -t bonzai` |
| `scp` path not found | Use full path `/leonardo_work/AIFAC_P02_222/...` not `$WORK` |
| `singularity build` OOM-killed | Use `--sandbox` not `.sif` |
| `python: not found` in container | Use `python3` |
| `mount source doesn't exist` | `mkdir -p` the bind-mount directories first |
| Model download fails on compute | Pre-download on login node, use local path |
| `Qwen3_5ForConditionalGeneration failed` | Need `transformers>=5.2.0` in container |
| `Failed to find C compiler` | Need gcc in container + libcuda stub symlink |
| Triton gcc linking fails | Need `libcuda.so.1` stub in container (see Dockerfile) |
| `input/output error` on docker pull | Disk full — run Recovery steps above |
| Docker prune says 0B reclaimed | Quit Docker Desktop, delete VM disk, restart |
| `vllm depends on transformers<5` | Use `--no-deps` override after vllm install |
| Docker pull times out on large image | Retry; layers are cached. Or use smaller base image |
