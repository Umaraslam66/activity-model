# Task: Run Gemma 4 on Leonardo for Synthetic Data Generation

## Read First

Before doing any work, read these files:

1. **`/SAT/claude.md`** — main operator guide and current project status
2. **`/SAT/docker/Dockerfile.qwen-generator`** — current container definition
3. **`/SAT/docker/README.md`** — container build and transfer workflow
4. **`/SAT/scripts/cineca/activity_chain_generate_leonardo.sbatch`** — main Leonardo generation job
5. **`/SAT/scripts/cineca/activity_chain_probe_leonardo.sbatch`** — Leonardo probe job
6. **`/SAT/activity_chain/backends.py`** — vLLM backend setup
7. **`/SAT/activity_chain/schema.py`** — output schema
8. **`/SAT/activity_chain/validation.py`** — post-generation validation
9. **`/SAT/requirements_inference.txt`** — runtime dependencies
10. **`/SAT/.github/workflows/build-container.yml`** — CI image build

## Project Goal

Bonzai is building an urban mobility simulator. This repository generates synthetic activity-chain data using a large teacher model.

Current phase:
- run `google/gemma-4-31B-it` on CINECA Leonardo A100 GPUs
- generate structured JSONL activity-chain records
- later train a smaller student model on that dataset

## Current State

### What works

- local CPU smoke tests
- CI image build
- Docker tar upload to Leonardo
- Leonardo sandbox build
- Leonardo GPU startup
- the previous Qwen3.5 runtime was stabilized
- the current target is Gemma 4
- vLLM on Leonardo when run with:
  - `trust_remote_code=True`
  - `limit_mm_per_prompt={image:0,audio:0}`
  - `enforce_eager=True`
  - `VLLM_USE_STANDALONE_COMPILE=0`

### Current blocker

The main blocker is no longer deployment.

The current blocker is **generation quality / validation**:
- the model produces JSON
- but many records violate chronology, indexing, or summary consistency
- Leonardo debug runs currently end up with all records rejected

## Infrastructure Constraints

- Leonardo is `x86_64`
- compute nodes have no internet
- use the local staged model path, not a Hugging Face model ID
- use `singularity build --sandbox`, not `.sif`
- keep `--cleanenv --nv`
- Gemma 4 31B should be treated as a 2-GPU tensor-parallel deployment on Leonardo
- use manual commands over wrapper scripts if reliability matters

## Success Criteria

Success means:

1. container and model run on Leonardo
2. probe passes
3. a debug generation job produces accepted JSONL records
4. reject rate is low enough to justify scaling out

## Research Guidance

If more investigation is needed, prioritize:

- official CINECA Leonardo / Singularity docs
- official vLLM docs
- official Gemma 4 model card
- actual vLLM source code when runtime behavior is unclear

Do not treat old failed paths as the source of truth. The surviving source of truth is `claude.md` plus the current runtime files in `scripts/` and `activity_chain/`.
