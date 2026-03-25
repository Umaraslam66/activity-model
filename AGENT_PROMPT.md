# Task: Deploy Qwen3.5-9B on CINECA Leonardo HPC for Synthetic Data Generation

## Who You Are Working With
I am a startup founder building an urban mobility simulator called Bonzai. I need your help deploying a model on a European supercomputer. I am NOT a DevOps expert — I need clear, tested commands that work the first time.

## Read These Files First (mandatory)
1. **`/SAT/CLAUDE.md`** — Complete project documentation: architecture, all issues encountered, what failed, progress log. READ THIS ENTIRELY before doing anything.
2. **`/SAT/commands.md`** — Command reference for the deployment pipeline.
3. **`/SAT/docker/Dockerfile.qwen-generator`** — Current container definition.
4. **`/SAT/scripts/cineca/activity_chain_generate_leonardo.sbatch`** — Slurm job script.
5. **`/SAT/activity_chain/backends.py`** — VllmBackend class that loads the model.
6. **`/SAT/requirements_inference.txt`** — Python dependencies.
7. **`/SAT/.github/workflows/build-container.yml`** — CI pipeline.

## What the Project Does
- We use **Qwen3.5-9B** (a multimodal LLM by Qwen) as a "teacher model" to generate 50k-100k synthetic activity-chain records (JSON) describing how urban residents move through their day.
- The model runs via **vLLM** on **NVIDIA A100 64GB GPUs** on CINECA Leonardo HPC (European supercomputer).
- We use **structured output** (JSON schema enforcement) via vLLM's `StructuredOutputsParams`.
- The code is written and tested — the ONLY problem is getting the container to work on Leonardo.

## The Infrastructure
- **Leonardo HPC**: Slurm scheduler, Singularity PRO 4.3.0 (not Docker), A100 GPUs, x86_64 architecture.
- **Compute nodes have NO internet** — model weights (19GB) are pre-downloaded at `$FAST/bonzai_cache/huggingface/Qwen--Qwen3.5-9B`.
- **Container workflow**: GitHub Actions CI builds `linux/amd64` Docker image → push to GHCR → pull on Mac → `docker save` → stream via SSH to Leonardo → `singularity build --sandbox`.
- **Singularity constraint**: Login nodes OOM-kill `.sif` builds — must use `--sandbox`.

## The Blocking Problem (where we are stuck)

When we submit a Slurm job, vLLM fails to load Qwen3.5-9B with this error:

```
Model architectures ['Qwen3_5ForConditionalGeneration'] failed to be inspected
```

The root cause chain:
1. **vLLM 0.18.0 pins `transformers<5`**, but Qwen3.5 architecture (`Qwen3_5ForConditionalGeneration`) requires `transformers>=5.2.0` for recognition.
2. We solved this by installing vllm first, then `pip install --no-deps transformers==5.3.0` to override.
3. But then **Triton JIT compilation fails** — vLLM uses Triton to compile GPU kernels at runtime. Triton calls gcc to compile `cuda_utils.c` which links against `-l:libcuda.so.1`. This linking fails inside the Singularity container.

The gcc error:
```
subprocess.CalledProcessError: Command '['/usr/bin/gcc', '/tmp/.../cuda_utils.c', '-O3', '-shared', '-fPIC', ..., '-l:libcuda.so.1', '-L/.singularity.d/libs', ...]' returned non-zero exit status 1.
```

The `-L/.singularity.d/libs` path should contain `libcuda.so.1` (bind-mounted from host via `--nv` flag), but with `--cleanenv` it's not found. We tried:
- `nvidia/cuda:12.4.1-runtime` base → no gcc, no CUDA headers
- Added gcc to runtime → gcc found but can't link libcuda
- `nvidia/cuda:12.4.1-devel` base → has CUDA stubs but linking still fails
- Symlinked CUDA stub to system lib path → currently testing
- `vllm/vllm-openai:v0.18.0` base → image too large (~15-20GB), can't pull reliably

## What I Need You To Do

1. **Research thoroughly** using official docs:
   - vLLM docs: https://docs.vllm.ai/en/stable/
   - vLLM Qwen3.5 recipe: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html
   - Qwen3.5-9B model card: https://huggingface.co/Qwen/Qwen3.5-9B
   - CINECA Singularity docs: https://docs.hpc.cineca.it/services/singularity.html
   - CINECA Leonardo docs: https://docs.hpc.cineca.it/hpc/leonardo.html
   - Triton docs on JIT compilation and CUDA requirements

2. **Find the correct way** to run vLLM 0.18.0 + Qwen3.5-9B inside a Singularity `--sandbox` container on an HPC with `--cleanenv --nv` flags. Specifically:
   - How to resolve the `transformers<5` vs `transformers>=5.2.0` conflict
   - How to make Triton JIT compilation work inside Singularity (the `libcuda.so.1` linking issue)
   - Whether there's a way to pre-compile Triton kernels during Docker build to avoid JIT at runtime
   - Whether there are env vars to disable Triton or use alternative backends

3. **Produce a working Dockerfile** and any necessary changes to the sbatch script.

4. **Test your assumptions** — don't guess. Check actual vLLM source code for how it handles model inspection, what triggers Triton, and whether there are bypass flags.

## Constraints
- Image must be `linux/amd64` and buildable in GitHub Actions (ubuntu-latest, ~14GB disk after cleanup)
- Image compressed tar must be <10GB (needs to be pulled on Mac with ~30GB free and uploaded over ~10Mbps)
- Must work with Singularity `--sandbox` (not `.sif`)
- Sbatch uses `--cleanenv --nv` (cleanenv to prevent PYTHONPATH leakage, --nv for GPU access)
- Python 3.10 in container
- Must use local model path (no internet on compute nodes)
- `language_model_only=True` must be set (Qwen3.5 is multimodal, we only need text)

## Where We Stand Right Now (2026-03-25)

**Read `CLAUDE.md` for the full progress log, all issues, and "What Failed" table.**

- The latest CI build is **green** on GitHub Actions (branch `activity-chain-phase-a`). It uses `nvidia/cuda:12.4.1-devel-ubuntu22.04` base with a `libcuda.so.1` stub symlink. This has NOT been tested on Leonardo yet.
- Model weights (19GB) are already on Leonardo at `$FAST/bonzai_cache/huggingface/Qwen--Qwen3.5-9B`.
- Code is already on Leonardo at `$WORK/bonzai/sentiment-action-transformer`.
- An old container sandbox exists on Leonardo but is stale — needs rebuild from the latest CI image.
- 0 core hours consumed out of 40,000 budget.

**The standard deploy cycle** (detailed in `commands.md`):
1. On Mac: `step ssh login` → `docker pull` → `docker save ... | ssh leonardo 'cat > .tar'`
2. On Leonardo (tmux): `singularity build --sandbox ... docker-archive://...tar`
3. On Mac: `rsync` code to Leonardo
4. On Leonardo: `export` env vars → `sbatch` the job
5. Check: `squeue`, `tail -f bonzai-qwen-gen-<JOBID>.out/.err`

**The untested fix**: The latest Dockerfile adds `ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1` so that Triton's gcc can link against it. This may or may not solve the problem. If it doesn't, a fundamentally different approach to the Triton/Singularity issue is needed.

**If the stub fix fails**, consider these alternative approaches (not yet tried):
- Pre-compile Triton kernels during Docker build (warm the cache)
- Set `VLLM_USE_V1=0` to use V0 engine (may avoid Triton during inspection)
- Set `VLLM_ATTENTION_BACKEND=FLASH_ATTN` to bypass Triton attention kernels
- Remove `--cleanenv` from sbatch and instead selectively `unset` problematic vars
- Use `vllm serve` as a server process instead of the Python API (different code path)

## What Success Looks Like
A submitted Slurm job that:
1. Loads Qwen3.5-9B via vLLM
2. Generates 100 JSON activity-chain records
3. Writes them to a JSONL file
4. No errors in stdout/stderr
