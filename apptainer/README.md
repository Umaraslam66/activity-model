# Apptainer / Singularity Workflow

## Purpose
The final HPC runtime artifact for CINECA / Leonardo is a `.sif` image pulled from GHCR, where it is built by GitHub Actions CI.

The image remains **deps-only**.
Model weights are downloaded or reused from persistent cache on the cluster.

## How the image is built

GitHub Actions builds the `linux/amd64` Docker image on every push and uploads it to:

```
ghcr.io/umaraslam66/bonzai-qwen-generator:<tag>
```

See `.github/workflows/build-container.yml` for the workflow definition.

## Pull the `.sif` on Leonardo

```bash
singularity pull --dir $WORK/containers \
  docker://ghcr.io/umaraslam66/bonzai-qwen-generator:activity-chain-phase-a
```

Rename for convenience:

```bash
mv $WORK/containers/bonzai-qwen-generator_activity-chain-phase-a.sif \
   $WORK/containers/bonzai-qwen-generator.sif
```

## Sanity check

```bash
singularity exec $WORK/containers/bonzai-qwen-generator.sif python3 --version
singularity exec $WORK/containers/bonzai-qwen-generator.sif python3 -c "import vllm; print(vllm.__version__)"
```

## Smoke test on Leonardo (CPU, no GPU needed)

```bash
singularity exec \
  --bind $WORK/bonzai/sentiment-action-transformer:/workspace \
  $WORK/containers/bonzai-qwen-generator.sif \
  python3 /workspace/scripts/run_activity_chain_smoke_test.py \
    --output-dir /workspace/data/apptainer-smoke
```

## Leonardo GPU usage

Pass the `.sif` to the Slurm wrapper:

```bash
sbatch \
  --export=ALL,PROJECT_ROOT=$WORK/bonzai/sentiment-action-transformer,ABM_LLM_MODEL=Qwen/Qwen3.5-9B,CONTAINER_IMAGE=$WORK/containers/bonzai-qwen-generator.sif,OUTPUT_ROOT=$WORK/bonzai/activity_chain_data,TOTAL_RECORDS=100,NUM_SHARDS=1,SHARD_ID=0 \
  $WORK/bonzai/sentiment-action-transformer/scripts/cineca/activity_chain_generate_leonardo.sbatch
```

## What NOT to do

- Do not build the image locally on Apple Silicon Mac (wrong arch + disk constraints).
- Do not use `singularity build --fakeroot` on Leonardo (account lacks subuid mapping).
- Do not bake model weights into the image.

## Promotion gates

1. CI build succeeds (green workflow).
2. `singularity pull` completes on Leonardo.
3. `python3 -c "import vllm"` works inside the `.sif`.
4. CPU smoke test passes inside the `.sif`.
5. First debug GPU shard (100 records) succeeds.
