# Leonardo Robust Deployment Plan

## Executive Summary

The current failures are not one bug. They are a compatibility-matrix problem across five layers:

1. **Model support:** Qwen3.5 is a very new architecture.
2. **Inference engine support:** the Qwen model card currently points to **vLLM main/nightly** and latest `transformers`, not an older released vLLM + manually overridden `transformers`.
3. **CUDA / driver support:** Leonardo official docs currently describe **NVIDIA driver 535.54.03 / CUDA 12.2** on A100 64GB nodes.
4. **Container runtime:** Triton JIT needs a real compiler + linker + CUDA-visible libraries inside Singularity `--nv`.
5. **Operational workflow:** mutable tags, stale sandboxes, and hour-long redeploy loops make every mismatch expensive.

The main strategic conclusion is:

- If we insist on **Qwen3.5-9B**, the robust path is to treat Leonardo as a **CUDA 12.2 target** and build / freeze a stack specifically for that target.
- If we want the **lowest-risk path to synthetic data generation**, we should be willing to switch to a model that is already supported by a released vLLM stack on Leonardo.

## What The Official Docs Imply

### Leonardo / CINECA

- Leonardo Booster nodes are **A100 64GB**.
- CINECA container docs currently describe Leonardo as **driver 535.54.03 / CUDA 12.2**.
- CINECA recommends `module load hpcx-mpi/2.19` and `module load cuda/12.2`.
- CINECA also notes that `SINGULARITYENV_LD_LIBRARY_PATH` should include the default Singularity GPU library path when overriding it.

### Singularity / GPU behavior

- `singularity exec --nv` bind-mounts host NVIDIA libraries and sets `LD_LIBRARY_PATH` inside the container for GPU apps.
- `--cleanenv` is still useful for avoiding host Python leakage, but it means we must be explicit about any environment variables we do rely on.

### vLLM

- Current official vLLM install docs say the prebuilt binaries are for newer CUDA variants and explicitly state that if you have a **different CUDA version**, you should **build vLLM from source**.
- Current official Qwen3.5 guidance says Qwen3.5 requires **vLLM from the main branch** and the **latest `transformers`** in a fresh environment.

This means the current repo strategy:

- `vllm==0.18.0`
- followed by `pip install --no-deps transformers==5.3.0`

is a workaround, not a supported matrix.

## Main Findings In This Repo

### 1. The docs disagree with the code

The repository currently has conflicting "source of truth" files:

- [`SAT/docker/Dockerfile.qwen-generator`](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/docker/Dockerfile.qwen-generator) uses `nvidia/cuda:12.4.1-devel-ubuntu22.04`.
- [`SAT/claude.md`](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/claude.md) says the current solution is `vllm/vllm-openai:v0.18.0`.
- [`SAT/docker/README.md`](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/docker/README.md) still documents `nvidia/cuda:12.4.1-runtime-ubuntu22.04`.
- [`SAT/apptainer/README.md`](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/apptainer/README.md) still assumes a direct `.sif` pull flow, while the current operational flow is Docker archive -> sandbox.

Before more debugging, the project needs a single deployment source of truth.

### 2. The sbatch script is still mixing host and container toolchains

[`SAT/scripts/cineca/activity_chain_generate_leonardo.sbatch`](/Users/umaraslam/Documents/dynamo/Bonzai/LTM/SAT/scripts/cineca/activity_chain_generate_leonardo.sbatch) currently bind-mounts:

- `/usr/bin/gcc`
- `/usr/bin/g++`

into the container.

That is high risk. Once the container already includes its own compiler and binutils, binding host compilers back into the container reintroduces the exact host/container mismatch we are trying to eliminate.

**Recommendation:** remove the gcc/g++ bind mounts and use the compiler that is baked into the container.

### 3. The current Dockerfile targets CUDA 12.4.1, but Leonardo is documented as CUDA 12.2

Even if this sometimes works, it is not the most robust target for Triton JIT on Leonardo. The closer the user-space CUDA stack is to the Leonardo host runtime, the fewer surprises we should expect.

### 4. The current workflow has no dedicated GPU preflight probe

Right now the first real GPU test is the full generation job. That is too expensive.

There should be a tiny Leonardo GPU probe that checks:

- `nvidia-smi`
- `python3 -c "import torch, transformers, vllm"`
- `ctypes.CDLL("libcuda.so.1")`
- `AutoConfig.from_pretrained(local_model_path)`
- `AutoTokenizer.from_pretrained(local_model_path)`
- a tiny Triton compile / cache write
- a tiny vLLM model construction step

The full 100-record generation job should only happen after this probe passes.

## Recommended Strategy

## Track A: Fastest path to a working Leonardo run

Use this if the immediate goal is: "generate data this week with minimum extra architecture changes."

1. **Stop treating mutable branch tags as deploy artifacts.**
   Use the image SHA / digest as the deployment identity, and version the remote sandbox path with that SHA.

2. **Keep `--cleanenv`, but explicitly export only the env we need.**
   Add explicit container env for:
   - `HF_HOME`
   - `HUGGINGFACE_HUB_CACHE`
   - `TRANSFORMERS_CACHE`
   - `VLLM_CACHE_ROOT`
   - `TRITON_CACHE_DIR`
   - `HF_HUB_OFFLINE=1`
   - `TRANSFORMERS_OFFLINE=1`
   - `TOKENIZERS_PARALLELISM=false`
   - `OMP_NUM_THREADS`
   - `LD_LIBRARY_PATH` including the Singularity GPU library path

3. **Remove host gcc/g++ binds.**
   The container must be self-sufficient.

4. **Persist Triton cache on `$FAST`.**
   This avoids recompiling kernels on every job and turns first-run JIT into a one-time cost.

5. **Split "GPU environment probe" from "generation job".**
   Probe first, then 1-record / 5-record smoke, then 100 records, then sharded production.

6. **Freeze the first known-good image and stop rebuilding it during data generation.**
   Once one image works, do not change packages again until the synthetic dataset job is complete.

## Track B: Robust long-term path for Qwen3.5

Use this if Qwen3.5 is non-negotiable and we want a supportable stack instead of a lucky workaround.

1. **Treat Leonardo CUDA 12.2 as the target ABI.**
2. **Build vLLM from source for that target** rather than relying on a wheel path optimized for newer CUDA variants.
3. **Pin by commit / digest**, not by floating release names.
4. **Generate and store a build manifest** in the image:
   - `python --version`
   - `pip freeze`
   - `torch.__version__`
   - `transformers.__version__`
   - `vllm.__version__`
   - CUDA toolkit version
   - gcc / g++ versions
5. **Run import-time and config-time smoke tests in CI** before pushing the image.

This is more work up front, but it is the right approach if Qwen3.5 remains the teacher model.

## Track C: Lowest-risk fallback if Leonardo keeps blocking Qwen3.5

If the objective is synthetic data, not specifically Qwen3.5 branding, the lowest-risk option is:

- switch to a released **text-only** Qwen model that is already supported by a released vLLM stack on Leonardo,
- keep `language_model_only` unnecessary by choosing a text-only architecture,
- finish data generation,
- return to Qwen3.5 only after the container toolchain is stable.

This is the least elegant option, but it may be the best business decision if time matters more than model novelty.

## Concrete Changes Recommended Next

### Container

- Pick one base strategy and document it everywhere.
- Prefer a **devel-style** base over a stripped runtime image if Triton JIT remains part of the path.
- Align the CUDA target with Leonardo as closely as possible.
- Bake in:
  - gcc
  - g++
  - python3-dev
  - the exact Python package set
- Emit a build manifest at image build time.

### Sbatch

- Remove bind mounts for host gcc / g++.
- Add `TRITON_CACHE_DIR=$FAST/bonzai_cache/triton`.
- Add offline env vars.
- Add explicit `SINGULARITYENV_LD_LIBRARY_PATH` including the default Singularity GPU libs path plus any CUDA stub path required by the image.
- Add a `PROBE_ONLY=1` mode or a separate probe script.

### CI

- Build one image per commit SHA.
- Run container smoke tests that assert:
  - imports of `torch`, `vllm`, `transformers`
  - expected package versions
  - `gcc --version`
  - presence of CUDA stubs / expected linker-visible files
  - `AutoConfig.from_pretrained()` on a small local fixture for Qwen3.5 config compatibility

### Operations

- Keep remote sandboxes versioned by image SHA.
- Keep a `current` symlink only after probe passes.
- Never overwrite the only working sandbox.

## Immediate Next Steps

1. Normalize the deployment docs so there is one real source of truth.
2. Edit the Leonardo sbatch wrapper to stop binding host gcc / g++ and to persist Triton cache.
3. Add a dedicated GPU probe job script.
4. Decide which strategic path we are taking:
   - keep Qwen3.5 and build for Leonardo properly,
   - or switch to a safer released text-only model for dataset generation.
5. Only after the probe passes, run the 100-record debug job.

## Sources

- CINECA Leonardo / Singularity docs:
  - https://docs.hpc.cineca.it/services/singularity.html
  - https://docs.hpc.cineca.it/hpc/leonardo.html
- Singularity GPU support:
  - https://docs.sylabs.io/guides/4.1/user-guide/gpu.html
- vLLM install docs:
  - https://docs.vllm.ai/en/stable/getting_started/installation/gpu/
- vLLM troubleshooting:
  - https://docs.vllm.ai/en/stable/getting_started/troubleshooting.html
- Qwen3.5 model card:
  - https://huggingface.co/Qwen/Qwen3.5-9B
