# Leonardo Operator Reference

Working notes and lessons for running `sentiment-action-transformer` on CINECA Leonardo.

## Allocation And Access

- project allocation: `40000 local core hours on Leonardo Booster`
- project window: `2026-03-11` to `2026-06-11`
- username: `uaslam00`
- login host: `login.leonardo.cineca.it`
- access method: CINECA `smallstep` SSH certificates with 2FA / OTP

Login flow:

```bash
eval "$(ssh-agent -s)"
step ssh login aslamumar16@gmail.com --provisioner cineca-hpc
ssh uaslam00@login.leonardo.cineca.it
```

Important:
- start `ssh-agent` before `step ssh login`
- if the agent is restarted, run `step ssh login` again

## Hardware And Queue

- target: Leonardo Booster
- GPUs: `4 x NVIDIA A100 64GB` per node
- CPUs: Intel Ice Lake (`x86_64`)
- production partition: `boost_usr_prod`
- production QOS: `normal`
- debug QOS: `boost_qos_dbg`
- container artifacts must be `linux/amd64`

References:
- https://docs.hpc.cineca.it/hpc/leonardo.html
- https://docs.hpc.cineca.it/services/singularity.html
- https://docs.hpc.cineca.it/general/access.html

## Storage Layout

- `$WORK=/leonardo_work/AIFAC_P02_222`
- `$FAST=/leonardo_scratch/fast/AIFAC_P02_222`

```
$WORK/bonzai/sentiment-action-transformer   # repo
$WORK/bonzai/activity_chain_data            # outputs
$WORK/containers                            # .sif images
$FAST/bonzai_cache                          # HF + vLLM cache
```

## Container Workflow (CI → GHCR → Leonardo)

The runtime image is built in GitHub Actions CI and pulled on Leonardo via Singularity.

### Why this path

- Local Mac cannot build large `linux/amd64` CUDA images (disk constraints, wrong arch)
- Leonardo `singularity build --fakeroot` is unavailable for this account
- Direct `pip install vllm` on Leonardo is fragile (PYTHONPATH leakage from cineca-ai)
- CI builds are reproducible, fast, and free

### Pull the image on Leonardo

```bash
singularity pull --dir $WORK/containers \
  docker://ghcr.io/umaraslam66/bonzai-qwen-generator:activity-chain-phase-a
mv $WORK/containers/bonzai-qwen-generator_activity-chain-phase-a.sif \
   $WORK/containers/bonzai-qwen-generator.sif
```

### Sanity check

```bash
singularity exec $WORK/containers/bonzai-qwen-generator.sif python3 --version
singularity exec $WORK/containers/bonzai-qwen-generator.sif python3 -c "import vllm; print(vllm.__version__)"
```

## What Worked

1. SSH access with `smallstep`
2. Repo upload with `rsync`
3. Local and Leonardo smoke tests (mock backend)
4. Registry-to-SIF pull (`singularity pull docker://...`)
5. Official CINECA AI stack (`profile/deeplrn` + `cineca-ai/4.3.0`) — exposes torch 2.2, CUDA 12.1

## What Failed (Do Not Repeat)

| Attempt | Why it failed |
|---------|--------------|
| Local Apptainer on macOS (Lima VM) | Unnecessary — Leonardo has Singularity |
| Default `arm64` Docker build from Mac | Leonardo is `x86_64` |
| Large local `linux/amd64` CUDA build | Mac disk too small |
| `singularity build --fakeroot` on Leonardo | Account lacks subuid mapping |
| `pip install vllm` in plain login-node venv | `CUDA_HOME` not set, source build attempted |
| `cineca-ai` + venv + vllm | `PYTHONPATH` leakage — imported wrong torch |
| `unset PYTHONPATH` + clean venv + vllm | Endless missing transitive deps |
| Reusing venv after switching Python modules | Stale interpreter in venv |

## Best Practices

1. **Use CI for container builds** — do not build locally on this Mac
2. **Use `singularity pull` on Leonardo** — do not `singularity build`
3. **Login nodes are setup nodes** — no GPUs, only prep work
4. **Load modern Python** — at minimum `module load python/3.11.7`
5. **Use `$FAST` for cache** — HF weights at `$FAST/bonzai_cache/huggingface`
6. **Keep debug runs small** — 100 records, 1 GPU, 1 shard first
7. **Always re-export SSH vars** in each terminal session
8. **Delete and recreate venvs** after changing Python modules

## Quick Session Checklist

### On Mac

```bash
export LEONARDO_USER=uaslam00
export LEONARDO_HOST=${LEONARDO_USER}@login.leonardo.cineca.it
eval "$(ssh-agent -s)"
step ssh login aslamumar16@gmail.com --provisioner cineca-hpc
ssh ${LEONARDO_HOST}
```

### On Leonardo

```bash
cd $WORK/bonzai/sentiment-action-transformer
```
