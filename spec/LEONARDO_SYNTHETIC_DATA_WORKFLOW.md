# Leonardo Synthetic Data Workflow

## Goal
The first production phase in this repository is **synthetic data generation** using a local Qwen 9B model on EuroHPC / CINECA GPU infrastructure.

For now, training is deferred.

The immediate objective is:
- validate the workflow locally without a GPU
- validate the runtime stack in a Docker-compatible image
- convert that image into Apptainer
- host Qwen 9B on Leonardo GPU nodes
- generate structured activity-chain JSONL records
- validate and store shard outputs safely
- scale generation horizontally across jobs

## Recommended Operating Model
Use **one Qwen 9B instance per GPU job**.

Do **not** start with multi-GPU tensor parallelism.

Reason:
- Qwen 9B fits comfortably on a single A100 64GB
- generation is embarrassingly parallel
- job-level sharding is simpler and more reliable than multi-GPU serving for this stage
- shard failures are easier to retry

## Infrastructure Layout
Recommended directory strategy on CINECA:

- repository checkout: `$WORK/bonzai/sentiment-action-transformer`
- model / tokenizer cache: `$FAST/bonzai_cache` if available, otherwise `$WORK/bonzai_cache`
- generated JSONL shards: `$WORK/bonzai/activity_chain_data`
- temporary scratch: `$SCRATCH/bonzai_tmp`

Do not rely on large node-local temporary storage on Leonardo Booster nodes.

## Runtime Recommendation
### Local
- lightweight Python environment for CPU smoke tests
- `mock` backend only

### Container source of truth
- Docker-compatible image built in GitHub Actions CI
- pushed to `ghcr.io/umaraslam66/bonzai-qwen-generator`
- contains repo code and inference dependencies (vllm, transformers, etc.)
- does not contain model weights

### HPC runtime
- Singularity `.sif` pulled from GHCR on Leonardo
- `singularity exec --nv` for GPU access
- repo code is bind-mounted into the container at runtime

Container-first is the only supported path. Direct pip installs on Leonardo are fragile.

## Execution Sequence
### 1. Smoke test locally on CPU
- run `scripts/run_activity_chain_smoke_test.py`
- verify the `mock` backend path
- verify schema generation and validation
- verify output path naming and shard IDs
- verify valid and reject files are both produced
- verify shard outputs are disjoint
- verify the first deterministic mock record matches `spec/fixtures/activity_chain_mock_expected_first_record.json`

### 2. Push to GitHub — CI builds the Docker image
- GitHub Actions builds `linux/amd64` image and pushes to GHCR
- see `.github/workflows/build-container.yml`

### 3. Pull `.sif` on Leonardo
- `singularity pull` from GHCR
- sanity check: `python3 --version`, `import vllm`
- container smoke test with mock backend

### 4. Run one short debug GPU job on Leonardo
Use a short Leonardo debug allocation to validate:
- model loads correctly
- tokenizer loads correctly
- vLLM starts
- one small generation run succeeds
- JSONL and rejection files are written where expected

### 5. Run single-GPU production shards
Once debug works, run multiple independent single-GPU jobs.

Each job should:
- use `--num-shards N`
- use its own `--shard-id`
- write to a unique output file

### 6. Merge and validate
After shard generation:
- concatenate valid JSONL shards
- retain rejection files
- run the validator across the full merged set
- produce a simple coverage report

### 7. Only then move to student-model training

## Why Sharding Matters
The generation script supports:
- `--shard-id`
- `--num-shards`

This means a global target such as `100000` records can be split across many jobs without changing the schema or prompt logic.

Example:
- `--total 100000 --num-shards 8 --shard-id 0`
- `--total 100000 --num-shards 8 --shard-id 1`
- ...
- `--total 100000 --num-shards 8 --shard-id 7`

Each job generates only its assigned subset of scenario IDs.

## Recommended First Production Run
Start with:
- 1 node
- 1 GPU
- 1 shard
- total target: `1000`
- batch size: `4` or `8`
- strict validation enabled

Then scale to:
- 8 or 16 shards
- one GPU per shard
- merged target `50000` to `100000`

## Operational Rules
- always keep valid and rejected outputs separate
- never overwrite shard files from different jobs
- use deterministic shard IDs
- keep the prompt, schema, and validator versioned together
- do not start student training until the synthetic dataset distribution looks reasonable

## Files Added For This Workflow
- `activity_chain/`
- `scripts/generate_activity_chains_qwen.py`
- `scripts/run_activity_chain_smoke_test.py`
- `docker/Dockerfile.qwen-generator`
- `docker/README.md`
- `apptainer/README.md`
- `scripts/cineca/activity_chain_generate_leonardo.sbatch`

## Immediate Next Actions
1. Install `requirements_smoke.txt` locally and run the CPU smoke test.
2. Build the Docker image and rerun the smoke test in the container.
3. Convert the Docker image to `.sif`.
4. Confirm the exact Leonardo account / partition / QoS names available to the project.
5. Run a 100-record debug shard.
6. Inspect valid vs rejected output ratio.
7. Adjust prompt and validation thresholds before large-scale generation.
