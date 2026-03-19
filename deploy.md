# Deploy: Leonardo Booster A100 Workflow

Step-by-step operator guide for running the synthetic activity-chain generation on CINECA Leonardo (Booster partition, A100 64GB GPUs).

Assumes:
- local machine is an Apple Silicon Mac (storage-constrained)
- container image is built in CI (GitHub Actions) and pushed to GHCR
- access is through CINECA 2FA + `smallstep`
- Leonardo username is `uaslam00`

See `leonardo.md` for hardware facts, lessons learned, and session checklists.

---

## Phase A: Local Validation (Mac)

### Step 1: Enter the repo

```bash
cd /Users/umaraslam/Documents/dynamo/Bonzai/LTM/sentiment-action-transformer
```

### Step 2: Run the local smoke test

```bash
.venv-codex/bin/python scripts/run_activity_chain_smoke_test.py
```

Expected output ends with `"status": "ok"`.

### Step 3: Push to GitHub to trigger CI build

```bash
git push origin activity-chain-phase-a
```

This triggers `.github/workflows/build-container.yml`, which:
- builds a `linux/amd64` Docker image from `docker/Dockerfile.qwen-generator`
- pushes it to `ghcr.io/umaraslam66/bonzai-qwen-generator`

### Step 4: Monitor the CI build

```bash
gh run list --workflow=build-container.yml --limit=5
gh run watch
```

Wait for the build to go green.

### Step 5: Make the GHCR package public (first time only)

Go to: https://github.com/users/Umaraslam66/packages/container/bonzai-qwen-generator/settings

Set visibility to **Public** so Leonardo can pull without authentication.

---

## Phase B: Leonardo Setup (one-time)

### Step 6: Set login variables and connect

```bash
export LEONARDO_USER=uaslam00
export LEONARDO_HOST=${LEONARDO_USER}@login.leonardo.cineca.it
eval "$(ssh-agent -s)"
step ssh login aslamumar16@gmail.com --provisioner cineca-hpc
ssh ${LEONARDO_HOST}
```

### Step 7: Create the directory layout

```bash
mkdir -p $WORK/bonzai/sentiment-action-transformer
mkdir -p $WORK/bonzai/activity_chain_data
mkdir -p $WORK/containers
mkdir -p ${FAST:-$WORK}/bonzai_cache
echo "WORK=$WORK  FAST=${FAST:-not_set}"
```

### Step 8: Upload the repo

From the **local Mac**, in the repo root:

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
  ./ ${LEONARDO_HOST}:\$WORK/bonzai/sentiment-action-transformer/
```

### Step 9: Verify Leonardo environment

On Leonardo:

```bash
which singularity || which apptainer
sinfo -p boost_usr_prod
saldo -b
```

Confirm: Singularity exists, partition is visible, budget is available.

### Step 10: Run code-only smoke test on Leonardo

```bash
cd $WORK/bonzai/sentiment-action-transformer
module load python/3.11.7
python3 -m venv .venv-smoke
source .venv-smoke/bin/activate
pip install -r requirements_smoke.txt
python scripts/run_activity_chain_smoke_test.py
```

Expected: `"status": "ok"`.

---

## Phase C: Container Deployment

### Step 11: Pull the CI-built image on Leonardo

```bash
singularity pull --dir $WORK/containers \
  docker://ghcr.io/umaraslam66/bonzai-qwen-generator:activity-chain-phase-a
```

Rename for convenience:

```bash
mv $WORK/containers/bonzai-qwen-generator_activity-chain-phase-a.sif \
   $WORK/containers/bonzai-qwen-generator.sif
```

### Step 12: Sanity check the `.sif`

```bash
singularity exec $WORK/containers/bonzai-qwen-generator.sif python3 --version
singularity exec $WORK/containers/bonzai-qwen-generator.sif python3 -c "import vllm; print(vllm.__version__)"
```

If `import vllm` fails, check the CI build logs — the image may need a dependency fix.

### Step 13: Container smoke test (CPU, no GPU needed)

```bash
singularity exec \
  --bind $WORK/bonzai/sentiment-action-transformer:/workspace \
  $WORK/containers/bonzai-qwen-generator.sif \
  python3 /workspace/scripts/run_activity_chain_smoke_test.py \
    --output-dir /workspace/data/container-smoke
```

Expected: `"status": "ok"`.

---

## Phase D: GPU Generation

### Step 14: Debug run — 100 records, 1 GPU

```bash
sbatch \
  --export=ALL,PROJECT_ROOT=$WORK/bonzai/sentiment-action-transformer,ABM_LLM_MODEL=Qwen/Qwen3.5-9B,CONTAINER_IMAGE=$WORK/containers/bonzai-qwen-generator.sif,OUTPUT_ROOT=$WORK/bonzai/activity_chain_data,TOTAL_RECORDS=100,NUM_SHARDS=1,SHARD_ID=0 \
  $WORK/bonzai/sentiment-action-transformer/scripts/cineca/activity_chain_generate_leonardo.sbatch
```

Monitor:

```bash
squeue -u $USER
tail -f bonzai-qwen-gen-*.out
tail -f bonzai-qwen-gen-*.err
```

### Step 15: Validate debug outputs

```bash
wc -l $WORK/bonzai/activity_chain_data/activity_chain_v1.shard0of1.jsonl
wc -l $WORK/bonzai/activity_chain_data/activity_chain_v1.shard0of1.rejects.jsonl
head -1 $WORK/bonzai/activity_chain_data/activity_chain_v1.shard0of1.jsonl | python3 -m json.tool
```

Check:
- valid records exist
- reject ratio is not extreme (< 30%)
- records contain `city_archetype`, `macro_context`, latent traits, and shocks

### Step 16: Production sharded run

After the debug shard looks good:

```bash
for i in $(seq 0 7); do
  sbatch \
    --export=ALL,PROJECT_ROOT=$WORK/bonzai/sentiment-action-transformer,ABM_LLM_MODEL=Qwen/Qwen3.5-9B,CONTAINER_IMAGE=$WORK/containers/bonzai-qwen-generator.sif,OUTPUT_ROOT=$WORK/bonzai/activity_chain_data,TOTAL_RECORDS=50000,NUM_SHARDS=8,SHARD_ID=$i \
    $WORK/bonzai/sentiment-action-transformer/scripts/cineca/activity_chain_generate_leonardo.sbatch
done
```

### Step 17: Merge and validate shards

```bash
cat $WORK/bonzai/activity_chain_data/activity_chain_v1.shard*of8.jsonl \
  > $WORK/bonzai/activity_chain_data/activity_chain_v1.merged.jsonl
wc -l $WORK/bonzai/activity_chain_data/activity_chain_v1.merged.jsonl
wc -l $WORK/bonzai/activity_chain_data/activity_chain_v1.shard*of8.rejects.jsonl
```

---

## Budget Notes

- allocation: `40,000 local core hours` on Leonardo Booster
- each GPU ≈ `1/4 node` for accounting
- production queue: `boost_usr_prod`, QOS: `normal`
- debug queue: `boost_usr_prod`, QOS: `boost_qos_dbg`

## When Something Fails

Debug in this order:
1. Can you SSH? → re-run `step ssh login`
2. Is Singularity available? → `which singularity`
3. Does the code smoke test pass? → `python scripts/run_activity_chain_smoke_test.py`
4. Can you pull any image? → `singularity pull docker://python:3.11-slim`
5. Does the real `.sif` import vllm? → `singularity exec ... python3 -c "import vllm"`
6. Does the sbatch run? → check `.out` and `.err` files

Always paste the full error before changing the workflow. See `leonardo.md` for known failure modes.

## Updating the Container

When code changes are pushed:

1. Push to GitHub → CI rebuilds automatically
2. On Leonardo: re-pull the `.sif`
3. Re-run rsync to update the repo code (the sbatch binds the repo into the container)

```bash
# From Mac:
rsync -avz \
  --exclude .git --exclude .venv --exclude .venv-codex --exclude .venv-smoke \
  --exclude __pycache__ --exclude .pytest_cache --exclude data --exclude '*.sif' \
  ./ ${LEONARDO_HOST}:\$WORK/bonzai/sentiment-action-transformer/
```
