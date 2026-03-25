# Bonzai Activity-Chain Model

## What This Project Is

Bonzai is an urban mobility simulator. This repository is the **model-side workspace** — the "brain" that generates synthetic training data.

**Core goal:** Given `State = Persona + Environment`, predict `Action = Activity Chain + Transport Mode`.

**Two-phase approach:**
1. **Phase 1 (current):** Use a large LLM (Qwen 3.5-9B) via vLLM on CINECA Leonardo A100 GPUs to generate 50k-100k synthetic full-day activity-chain records. This is the **teacher model** generating training data.
2. **Phase 2 (future):** Train a small, fast **student model** on that synthetic data so it can run in real-time inside a C++ transport simulator — no LLM needed at runtime.

The C++/CUDA routing engine, simulator runtime, and transport-network execution code live in a **separate repository** and are intentionally not present here.

## Current Project Status (2026-03-25)

**What works:**
- Code is written, tested, and passing smoke tests locally and on Leonardo
- CI pipeline builds `linux/amd64` Docker image from `vllm/vllm-openai:v0.18.0` base, pushes to GHCR
- SSH access to Leonardo confirmed, budget active (40k core hours, 0 consumed, until 2026-06-11)
- Repo uploaded to Leonardo via rsync
- Qwen3.5-9B model weights pre-downloaded on Leonardo at `$FAST/bonzai_cache/huggingface/Qwen--Qwen3.5-9B` (19GB)
- Container sandbox built on Leonardo with vLLM 0.18.0
- `language_model_only=True` set in VllmBackend (Qwen3.5 is natively multimodal, we only need text)
- Sbatch script has gcc/g++ bind-mounts and `SINGULARITYENV_CC` for Triton JIT

**Current step:** Waiting for CI build with `vllm/vllm-openai:v0.18.0` base image + `transformers==5.3.0`. Once green: pull on Mac → stream to Leonardo → rebuild sandbox → submit debug job.

**Key dependency chain solved:**
- Qwen3.5-9B needs `transformers>=5.2.0` for architecture recognition
- vLLM 0.18.0 pins `transformers<5` — conflict resolved by `pip install --no-deps transformers==5.3.0`
- Triton JIT needs gcc + CUDA libs with correct paths — resolved by using official vLLM base image
- Qwen3.5 is multimodal — `language_model_only=True` skips vision encoder for text-only use

**What is not implemented yet:**
- Student-model training pipeline
- Feature flattening for supervised training
- ONNX export
- Integration with the separate C++ engine repo

## Directory Map

```
activity_chain/                    # Main Python package (1093 LOC)
  schema.py                        # JSON schema, enums, city archetypes, system prompt (354 LOC)
  seeds.py                         # Monte Carlo persona/environment generation (172 LOC)
  backends.py                      # MockBackend + VllmBackend (302 LOC)
  validation.py                    # Schema & plausibility checks (125 LOC)
  prompting.py                     # Chat prompt assembly (31 LOC)
  runner.py                        # Generation loop, file writing, reject handling (108 LOC)

scripts/
  generate_activity_chains_qwen.py # Main CLI entrypoint
  run_activity_chain_smoke_test.py # Local CPU smoke test
  cineca/
    activity_chain_generate_leonardo.sbatch  # Leonardo Slurm wrapper

spec/
  ACTIVITY_CHAIN_V1.md             # High-level record specification
  LEONARDO_SYNTHETIC_DATA_WORKFLOW.md
  fixtures/
    activity_chain_mock_expected_first_record.json  # Deterministic test fixture

docker/
  Dockerfile.qwen-generator        # CUDA 12.4.1-runtime base
  README.md

apptainer/
  README.md                        # Singularity/.sif notes for Leonardo

.github/workflows/
  build-container.yml              # CI: builds linux/amd64, pushes to GHCR

requirements_smoke.txt             # jsonschema, pytest (local dev)
requirements_inference.txt         # vllm, transformers, etc. (GPU runtime)
```

**Rules:**
- Business logic belongs in `activity_chain/`, scripts stay thin
- `spec/` explains the system, no legacy workflows
- Keep `.dockerignore` tight to avoid bloating the image

## Data Model

Each generated JSONL record contains:

**State (input):**
- `city_archetype`: `dense_transit_hub`, `sprawling_car_city`, `winter_cycling_city`
- `macro_context`: `transit_density_index` (0-1), `active_mobility_index` (0-1), `spatial_sprawl_score` (0-1)
- `persona`: age, income, employment, household, car/bike/transit ownership, work_schedule, plus latent traits:
  - `value_of_time_multiplier` (0.5-2.0)
  - `weather_resilience` (0-1)
  - `comfort_preference` (0-1)
- `environment`: day_type, weather, transit_disruption, congestion, parking/toll costs, special_event, plus shocks:
  - `dynamic_cost_sek` (0-500)
  - `network_delay_mins` (0-60)

**Action (output):**
- `activities`: 2-12 activities per day, all within 0-1440 minutes
- `legs`: travel segments connecting activities (mode, distance, duration)
- `summary`: primary_tour_purpose, commute_mode, distance/time totals

## Runtime Modes

### Local CPU smoke (mock backend)
```bash
python scripts/run_activity_chain_smoke_test.py
```
No GPU, no model weights, no vLLM needed. Tests schema flow, validation, file output, shard partitioning. Mock backend generates deterministic records with intentional errors (every 4th record invalid).

### Real generation (vllm backend)
Runs `Qwen/Qwen3.5-9B` via vLLM on A100 GPUs. Sampling: temperature=0.8, top_p=0.95, max_tokens=2200. vLLM enforces JSON schema compliance via structured outputs.

---

## Leonardo HPC: Complete Reference

### Access

| Field | Value |
|-------|-------|
| Login host | `login.leonardo.cineca.it` |
| Username | `uaslam00` |
| Auth | CINECA 2FA with `smallstep` SSH certificates |
| Allocation | 40,000 local core hours on Leonardo Booster |
| Project window | 2026-03-11 to 2026-06-11 |
| Account code | `AIFAC_P02_222` |

### SSH Login Flow (from Mac)

```bash
export LEONARDO_USER=uaslam00
export LEONARDO_HOST=${LEONARDO_USER}@login.leonardo.cineca.it
eval "$(ssh-agent -s)"
step ssh login aslamumar16@gmail.com --provisioner cineca-hpc
ssh ${LEONARDO_HOST}
```

**Important:**
- Start `ssh-agent` BEFORE `step ssh login`
- Certificates expire — if you get `Permission denied (publickey)`, re-run `step ssh login`
- If the agent is restarted, run `step ssh login` again

### Hardware

| Field | Value |
|-------|-------|
| Partition | `boost_usr_prod` |
| QOS | `normal` (production), `boost_qos_dbg` (debug) |
| GPUs | 4x NVIDIA A100 64GB per node |
| CPUs | Intel Ice Lake (x86_64) |
| Container artifacts | Must be `linux/amd64` |

### Storage Layout

```
$WORK = /leonardo_work/AIFAC_P02_222
$FAST = /leonardo_scratch/fast/AIFAC_P02_222

$WORK/bonzai/sentiment-action-transformer   # repo checkout
$WORK/bonzai/activity_chain_data            # generation outputs
$WORK/containers/bonzai-qwen-generator      # container (sandbox directory)
$WORK/containers/bonzai-qwen-generator.tar  # Docker archive (can delete after sandbox built)
$FAST/bonzai_cache/huggingface              # HF model weights cache
$FAST/bonzai_cache/vllm                     # vLLM cache
```

### Key Constraints

- **Login nodes have NO GPUs** — only setup, file transfers, container prep
- **Login nodes have strict memory limits** — large squashfs compression gets OOM-killed
- **Compute nodes have NO internet** — containers and model weights must be pre-staged
- **System Python is 3.6.8** — always `module load python/3.11.7` first
- **No sudo** — cannot run `singularity build` from definition files
- **No `--fakeroot`** — account lacks subuid mapping
- **`$FAST` scratch** is auto-cleaned after 40 days

### References

- CINECA Leonardo guide: https://docs.hpc.cineca.it/hpc/leonardo.html
- CINECA access guide: https://docs.hpc.cineca.it/general/access.html
- CINECA Singularity guide: https://docs.hpc.cineca.it/services/singularity.html

---

## Container Workflow

### Architecture

```
Mac (git push) → GitHub Actions CI → GHCR → Mac (docker pull + save) → scp → Leonardo (singularity build --sandbox)
```

### Why containers (not pip install on Leonardo)

Every attempt to install vLLM directly on Leonardo failed:
- `pip install vllm` → `CUDA_HOME` not set, triggers source build
- `cineca-ai` module + venv → `PYTHONPATH` leakage imports wrong torch
- `unset PYTHONPATH` + clean venv → endless missing transitive deps
- Reusing venv after switching Python modules → stale interpreter

The container bundles Python, vLLM, torch, and all dependencies into one portable artifact.

### How the container was built and deployed

**Step 1: CI builds the Docker image (already done)**

GitHub Actions workflow (`.github/workflows/build-container.yml`) triggers on push to `activity-chain-phase-a` or `main`. Builds `linux/amd64` image from `docker/Dockerfile.qwen-generator` (based on `vllm/vllm-openai:v0.18.0`), pushes to `ghcr.io/umaraslam66/bonzai-qwen-generator`. CI uses `no-cache: true` to avoid stale layer issues.

**Step 2: Pull image on Mac and save as tar**

```bash
docker pull --platform linux/amd64 ghcr.io/umaraslam66/bonzai-qwen-generator:activity-chain-phase-a
docker save ghcr.io/umaraslam66/bonzai-qwen-generator:activity-chain-phase-a > /tmp/bonzai-qwen-generator.tar
```

**Step 3: Upload tar to Leonardo**

```bash
scp /tmp/bonzai-qwen-generator.tar ${LEONARDO_HOST}:/leonardo_work/AIFAC_P02_222/containers/
```

Note: Use the full path `/leonardo_work/AIFAC_P02_222/...` — `$WORK` does not expand over scp.

**Step 4: Build sandbox on Leonardo**

```bash
export SINGULARITY_TMPDIR=$WORK/tmp
mkdir -p $SINGULARITY_TMPDIR
singularity build --sandbox $WORK/containers/bonzai-qwen-generator docker-archive://$WORK/containers/bonzai-qwen-generator.tar
```

**CRITICAL: Use `--sandbox`, not a plain `singularity build` to `.sif`.** The squashfs compression step for `.sif` gets OOM-killed on login nodes every time. Sandbox creates a directory instead — uses less memory, works reliably. Singularity can exec from a sandbox directory the same way as a `.sif`.

**Step 5: Verify**

```bash
singularity exec $WORK/containers/bonzai-qwen-generator python3 --version
# Expected: Python 3.10.12

singularity exec $WORK/containers/bonzai-qwen-generator python3 -c "import vllm; print(vllm.__version__)"
# Expected: 0.18.0 (libcuda warning is normal on login node — no GPU)
```

### Container update workflow

When code changes require a new container:
1. Push to `activity-chain-phase-a` branch → CI builds new image
2. On Mac: `docker pull`, `docker save`, `scp` to Leonardo
3. On Leonardo: delete old sandbox, rebuild with `singularity build --sandbox`

---

## Running Jobs on Leonardo

### Upload repo code

From Mac, in the SAT directory:

```bash
rsync -avz --exclude .git --exclude .venv --exclude .venv-codex --exclude __pycache__ --exclude .pytest_cache --exclude data --exclude '*.sif' ./ ${LEONARDO_HOST}:/leonardo_work/AIFAC_P02_222/bonzai/sentiment-action-transformer/
```

### Smoke test on Leonardo (no GPU)

```bash
module load python/3.11.7
cd $WORK/bonzai/sentiment-action-transformer
python3 -m venv .venv-smoke
source .venv-smoke/bin/activate
pip install --upgrade pip
pip install -r requirements_smoke.txt
python scripts/run_activity_chain_smoke_test.py
# Expected: {"status": "ok", ...}
```

### Pre-flight: ensure cache directories exist

```bash
mkdir -p $FAST/bonzai_cache/huggingface
mkdir -p $FAST/bonzai_cache/vllm
```

The sbatch script bind-mounts these into the container. If they don't exist, Singularity will fail with `mount source doesn't exist`.

### Pre-flight: download model weights on login node

Compute nodes have NO internet access. Model weights must be pre-downloaded on the login node.

```bash
tmux new -s download
module load python/3.11.7
source .venv-smoke/bin/activate
pip install huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download Qwen/Qwen3.5-9B --local-dir $FAST/bonzai_cache/huggingface/Qwen--Qwen3.5-9B
```

Ctrl+B, D to detach. The download is ~18GB. Check progress with `tmux attach -t download`.

### Debug run (100 records, 1 GPU)

Use the **local path** to the model weights, not the HuggingFace ID.

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

**Important:** Export variables individually then run `sbatch` on one line. Long `--export=` strings break when the terminal wraps and inserts spaces.

### Monitor

```bash
squeue -u $USER
tail -f bonzai-qwen-gen-*.out
tail -f bonzai-qwen-gen-*.err
```

### Validate outputs

```bash
wc -l $WORK/bonzai/activity_chain_data/activity_chain_v1.shard0of1.jsonl
wc -l $WORK/bonzai/activity_chain_data/activity_chain_v1.shard0of1.rejects.jsonl
head -1 $WORK/bonzai/activity_chain_data/activity_chain_v1.shard0of1.jsonl | python3 -m json.tool
```

Check: valid records exist, reject count is not extreme, records include `city_archetype`, `macro_context`, latent traits, and shocks.

### Production sharded run (50k records)

```bash
export PROJECT_ROOT=$WORK/bonzai/sentiment-action-transformer
export ABM_LLM_MODEL=Qwen/Qwen3.5-9B
export CONTAINER_IMAGE=$WORK/containers/bonzai-qwen-generator
export OUTPUT_ROOT=$WORK/bonzai/activity_chain_data
export TOTAL_RECORDS=50000
export NUM_SHARDS=8
for i in $(seq 0 7); do
  export SHARD_ID=$i
  sbatch scripts/cineca/activity_chain_generate_leonardo.sbatch
done
```

### Budget

- Allocation: 40,000 local core hours
- Each GPU = 1/4 node for accounting
- Debug run (100 records) uses minimal budget — always test small first

---

## Sbatch Script Details

File: `scripts/cineca/activity_chain_generate_leonardo.sbatch`

The script:
1. Validates required env vars: `PROJECT_ROOT`, `CONTAINER_IMAGE`, `ABM_LLM_MODEL`, `OUTPUT_ROOT`, `NUM_SHARDS`, `SHARD_ID`
2. Sets up cache paths (prefers `$FAST`, falls back to `$WORK`)
3. Exports `HF_HOME`, `VLLM_CACHE_ROOT`, `TOKENIZERS_PARALLELISM=false`
4. Runs `module purge`, `module load hpcx-mpi/2.19`, `module load cuda/12.2`
5. Unsets `PYTHONPATH` and `PYTHONUSERBASE` (prevents leakage)
6. Detects singularity or apptainer
7. Accepts both `.sif` files and sandbox directories as `CONTAINER_IMAGE`
8. Runs `srun singularity exec --cleanenv --nv` with bind mounts for project, output, HF cache, and vLLM cache

Configurable env vars (with defaults):
- `BATCH_SIZE` (4), `MAX_TOKENS` (2200), `MAX_MODEL_LEN` (8192)
- `TEMPERATURE` (0.8), `TOP_P` (0.95), `DTYPE` (bfloat16)
- `TENSOR_PARALLEL_SIZE` (1), `ARCHETYPES` (empty = all)

---

## Issues Encountered and Solutions

### Issue: `singularity pull` / `singularity build` OOM-killed on login node
**Symptom:** `signal: killed` during "Creating SIF file" step.
**Root cause:** Login nodes have strict memory limits. Squashfs compression of ~5GB images exceeds them.
**Solution:** Use `singularity build --sandbox` instead. Creates an uncompressed directory, uses much less memory. Works identically for `singularity exec`.

### Issue: `singularity pull` on compute nodes fails with "network unreachable"
**Symptom:** `dial tcp: connect: network is unreachable`
**Root cause:** Compute nodes have no internet access.
**Solution:** All container pulls and model weight downloads must happen on login nodes (or pre-stage from Mac).

### Issue: `singularity build` from `.tar.gz` fails with "invalid tar header"
**Symptom:** `archive/tar: invalid tar header`
**Root cause:** Singularity's `docker-archive://` handler may not support gzip.
**Solution:** Gunzip first (`gunzip file.tar.gz`), then `singularity build ... docker-archive://file.tar`.

### Issue: Python 3.6 on Leonardo
**Symptom:** `SyntaxError: future feature annotations is not defined`
**Root cause:** System Python is 3.6.8.
**Solution:** Always run `module load python/3.11.7` before creating venvs.

### Issue: `pip install vllm` on Leonardo fails
**Symptom:** Attempts to compile from source, missing CUDA_HOME.
**Root cause:** Login nodes don't have full CUDA dev environment.
**Solution:** Don't install vllm on Leonardo. Use the container.

### Issue: PYTHONPATH leakage from cineca-ai module
**Symptom:** Wrong torch version imported, import errors.
**Root cause:** `cineca-ai` module sets PYTHONPATH that conflicts with venv.
**Solution:** `unset PYTHONPATH` + `--cleanenv` flag on singularity. The sbatch script handles this.

### Issue: Singularity mount fails for non-existent directories
**Symptom:** `mount source /path/to/dir doesn't exist`
**Root cause:** `--bind` requires the host path to exist.
**Solution:** Pre-create all bind-mount directories: `mkdir -p $FAST/bonzai_cache/huggingface $FAST/bonzai_cache/vllm`

### Issue: SSH drops during long operations
**Symptom:** `Connection closed by remote host`, `Broken pipe`
**Solution:** Use `tmux new -s <name>` for long operations. Detach with Ctrl+B, D. Reattach with `tmux attach -t <name>`. Alternative: `nohup <cmd> > $WORK/log.txt 2>&1 &`

### Issue: scp with `$WORK` fails
**Symptom:** `No such file or directory`
**Root cause:** `$WORK` is a Leonardo env var, not expanded on Mac.
**Solution:** Use the full path: `scp file ${LEONARDO_HOST}:/leonardo_work/AIFAC_P02_222/containers/`

### Issue: sbatch `--export` string breaks on paste
**Symptom:** `Unable to open file` error with partial path.
**Root cause:** Long single-line commands get split by terminal line wrapping, inserting spaces in the middle of paths.
**Solution:** Export variables individually, then run `sbatch scripts/cineca/activity_chain_generate_leonardo.sbatch` alone.

### Issue: Qwen3.5-9B requires vLLM >= 0.17.0
**Symptom:** Model fails to load or unrecognized architecture errors.
**Root cause:** Qwen3.5 model family (with Gated Delta Networks + sparse MoE) was first supported in vLLM v0.17.0.
**Solution:** Pin `vllm>=0.18.0` in `requirements_inference.txt`. Rebuild container.

### Issue: Compute nodes cannot download model weights
**Symptom:** `httpx.ConnectError: [Errno 101] Network is unreachable` when vLLM tries to fetch model config from HuggingFace.
**Root cause:** Compute nodes have no internet access (confirmed by curl test).
**Solution:** Pre-download model weights on login node using `huggingface-cli download`, then pass the local path as `ABM_LLM_MODEL` instead of the HuggingFace ID.

### Issue: Container has `python3` but not `python`
**Symptom:** `"python": executable file not found in $PATH`
**Root cause:** Ubuntu 22.04 base image only provides `python3`.
**Solution:** Use `python3` in the sbatch ARGS array, not `python`.

### Issue: Apptainer not installable on macOS
**Symptom:** `brew install apptainer` → `Linux is required for this software`
**Root cause:** Apptainer/Singularity only runs on Linux.
**Solution:** Don't try to build `.sif` on Mac. Use `docker save` → `scp` → `singularity build --sandbox` on Leonardo.

### Issue: Triton JIT compilation needs C compiler inside container
**Symptom:** `RuntimeError: Failed to find C compiler. Please specify via CC environment variable.`
**Root cause:** vLLM 0.18.0 uses Triton for GPU kernel compilation at runtime, which requires gcc.
**Solution (partial):** Added `gcc g++ python3-dev` to Dockerfile. But gcc still failed to link `libcuda.so.1` when using `nvidia/cuda:12.4.1-runtime` base + Singularity `--cleanenv --nv`. The `--cleanenv` flag strips library paths that `--nv` normally sets.
**Final solution:** Switched to `vllm/vllm-openai:v0.18.0` base image which has gcc, CUDA, and all library paths pre-configured.

### Issue: Triton gcc linking fails against libcuda.so.1 in Singularity
**Symptom:** `subprocess.CalledProcessError: Command '['/usr/bin/gcc', ... '-l:libcuda.so.1', '-L/.singularity.d/libs', ...]' returned non-zero exit status 1.`
**Root cause:** Singularity `--nv` bind-mounts host NVIDIA libs to `/.singularity.d/libs`, but `--cleanenv` strips the `LD_LIBRARY_PATH` that gcc/linker needs. The `-L/.singularity.d/libs` flag was passed but the library still wasn't found (possibly missing stubs in runtime image).
**Solution:** Use `vllm/vllm-openai:v0.18.0` base image which includes all CUDA development files with correct paths.

### Issue: vLLM 0.18.0 fails to inspect Qwen3.5 architecture
**Symptom:** `Model architectures ['Qwen3_5ForConditionalGeneration'] failed to be inspected`
**Root cause:** vLLM 0.18.0 has Qwen3.5 model code, but the model inspection subprocess uses `transformers` to parse the config. vLLM 0.18.0 pins `transformers<5` which installed 4.57.6. Qwen3.5 architecture recognition requires `transformers>=5.2.0`.
**Solution:** Install vllm first (gets transformers 4.x), then `pip install --no-deps transformers==5.3.0` to override without triggering pip resolver conflicts. Also add `language_model_only=True` to VllmBackend since Qwen3.5 is natively multimodal.

### Issue: vllm 0.18.0 pins transformers<5 but Qwen3.5 needs >=5.2.0
**Symptom:** `ERROR: Cannot install transformers==5.3.0 ... vllm 0.18.0 depends on transformers<5`
**Root cause:** Pip dependency resolver won't allow both constraints simultaneously.
**Solution:** Two-stage install in Dockerfile: first `pip install vllm==0.18.0` (gets transformers 4.x), then `pip install --no-deps transformers==5.3.0 huggingface_hub==1.7.2 accelerate==1.13.0` to override.

### Issue: Mac out of disk space during docker pull
**Symptom:** `input/output error` / `ENOSPC` when pulling ~8GB image.
**Root cause:** Mac disk had <3GB free, image requires ~8-9GB for pull + save.
**Solution:** Run `docker system prune -af && docker volume prune -af` to reclaim Docker cache space, then retry pull.

---

## What Failed (Do Not Repeat)

| Attempt | Why it failed |
|---------|--------------|
| Local Apptainer on macOS (Lima VM) | Unnecessary — Leonardo has Singularity |
| Default `arm64` Docker build from Mac | Leonardo is `x86_64` |
| Large local `linux/amd64` CUDA build | Mac disk too small |
| `singularity build --fakeroot` on Leonardo | Account lacks subuid mapping |
| `singularity pull` large image on login node | OOM-killed during squashfs compression |
| `singularity build` to `.sif` on login node | Same OOM issue |
| `pip install vllm` in plain login-node venv | `CUDA_HOME` not set, source build attempted |
| `cineca-ai` + venv + vllm | `PYTHONPATH` leakage — imported wrong torch |
| `unset PYTHONPATH` + clean venv + vllm | Endless missing transitive deps |
| Reusing venv after switching Python modules | Stale interpreter in venv |
| `brew install apptainer` on Mac | Linux-only software |
| vLLM 0.8.5 with Qwen3.5-9B | Qwen3.5 requires vLLM >= 0.17.0 |
| Using HuggingFace model ID on compute node | No internet — must use local path |
| `python` in sbatch ARGS | Container only has `python3` |
| vLLM 0.18.0 without gcc in container | Triton JIT needs C compiler at runtime |
| vLLM 0.18.0 with transformers 4.x | Qwen3.5 needs transformers >= 5.2.0 |
| `nvidia/cuda:12.4.1-runtime` base + gcc + `--cleanenv --nv` | Triton gcc can't link libcuda.so.1 — library paths stripped by `--cleanenv` |
| `pip install transformers==5.3.0 vllm==0.18.0` together | vllm pins `transformers<5` — must use `--no-deps` override |
| GHA Docker build cache (`cache-from: type=gha`) | Stale apt layer didn't include gcc — use `no-cache: true` |
| Docker prune without restarting Docker Desktop | VM disk doesn't shrink — must restart Docker Desktop to compact |

---

## Best Practices

1. **Use CI for container builds** — do not build locally on Mac
2. **Use `singularity build --sandbox`** — not `.sif` (login node OOM)
3. **Use `tmux`** for any operation that takes more than 1 minute on Leonardo
4. **Load modern Python** — `module load python/3.11.7` before anything
5. **Use `$FAST` for cache** — HF weights at `$FAST/bonzai_cache/huggingface`
6. **Pre-create all bind-mount directories** before submitting sbatch jobs
7. **Export sbatch variables individually** — don't use long `--export=` strings
8. **Keep debug runs small** — 100 records, 1 GPU, 1 shard first
9. **Always re-run `step ssh login`** if you get Permission denied
10. **Delete and recreate venvs** after changing Python modules
11. **Use full paths in scp** — `$WORK` doesn't expand on Mac
12. **Use `vllm/vllm-openai` as Docker base** — not bare `nvidia/cuda` runtime (missing dev files for Triton JIT)
13. **Override transformers with `--no-deps`** — vLLM pins `transformers<5` but Qwen3.5 needs `>=5.2.0`
14. **Restart Docker Desktop to compact VM disk** — `docker system prune` alone doesn't shrink the VM

---

## Next Work Items

1. ~~Run local smoke test~~ DONE
2. ~~Set up GitHub Actions CI~~ DONE
3. ~~Push branch to trigger CI build~~ DONE
4. ~~Upload repo to Leonardo~~ DONE
5. ~~Leonardo smoke test (mock backend)~~ DONE
6. ~~Build container on Leonardo (sandbox)~~ DONE (rebuilt multiple times)
7. ~~Verify container (python, vllm import)~~ DONE — vLLM 0.18.0 confirmed
8. ~~Create cache directories~~ DONE
9. ~~Discover vLLM 0.8.5 incompatible with Qwen3.5~~ DONE — upgraded to 0.18.0
10. ~~Upgrade to vLLM 0.18.0 + add gcc to Dockerfile~~ DONE
11. ~~Download Qwen3.5-9B weights on Leonardo~~ DONE — 19GB at `$FAST/bonzai_cache/huggingface/Qwen--Qwen3.5-9B`
12. ~~Discover transformers 4.x can't recognize Qwen3.5~~ DONE — need transformers>=5.2.0
13. ~~Discover Triton gcc linking fails with runtime CUDA base~~ DONE
14. ~~Switch to `vllm/vllm-openai:v0.18.0` base image~~ DONE — CI building
15. **NEXT:** Wait for CI build, pull on Mac, stream to Leonardo, rebuild sandbox
16. Rsync latest code to Leonardo
17. Re-run debug sbatch (100 records, 1 GPU) with local model path
18. Inspect valid/reject ratio and archetype distribution
19. Run sharded production generation (50k-100k records)
20. Begin student-model dataset flattening

---

## Progress Log

### 2026-03-24
- Built container as sandbox on Leonardo (login node OOM prevents `.sif` creation).
- Container verified: Python 3.10.12, vLLM 0.8.5.post1.
- First sbatch job (38371386) failed: `$FAST/bonzai_cache/huggingface` directory did not exist. Fix: pre-create bind-mount directories.
- Second sbatch job (38371474) failed: `python` not found in container. Fix: use `python3` in sbatch ARGS.
- Third sbatch job (38371872) failed: `StructuredOutputsParams` import error (vLLM 0.8.5 uses `GuidedDecodingParams`). Fixed temporarily, then discovered real issue.
- Discovered vLLM 0.8.5 does NOT support Qwen3.5-9B at all — requires vLLM >= 0.17.0.
- Upgraded `requirements_inference.txt` to `vllm>=0.18.0`, pushed to trigger CI rebuild.
- Confirmed compute nodes have no internet (curl test returned 000). Model weights must be pre-downloaded on login node.
- vLLM 0.18.0 uses `StructuredOutputsParams` (same as original code), so no code changes needed.
- Updated sbatch script to accept both `.sif` files and sandbox directories (`-f` check → `-f || -d`).
- Consolidated `deploy.md` and `leonardo.md` into this file.
- Built new sandbox with vLLM 0.18.0, submitted job 38379737 — failed with `GuidedDecodingParams` error (stale sed edit on Leonardo). Fixed with sed back to `StructuredOutputsParams`.
- Job 38379814: `RuntimeError: Failed to find C compiler` — Triton JIT needs gcc.
- Added `gcc g++ python3-dev` to Dockerfile, pushed to trigger CI rebuild.
- CI build succeeded (run 23509183430).
- Downloaded Qwen3.5-9B weights on Leonardo login node (19GB in 10 seconds to `$FAST`).
- Cleaned old sandbox and tar from Leonardo.
- Mac ran out of disk space during `docker pull` (~3GB free, need ~9GB).

### 2026-03-25
- Cleaned Mac Docker (removed redis, postgis images + volumes, freed ~2GB).
- Docker pull still failed (ENOSPC) — Docker Desktop VM held 15GB of stale data.
- Restarted Docker Desktop to compact VM disk — freed up to 34GB.
- Pulled new image (vLLM 0.18.0 + transformers 4.57.6 + gcc), verified gcc present.
- Streamed to Leonardo via `docker save | ssh`, built sandbox. vLLM 0.18.0 confirmed.
- Job 38380850 failed: gcc bind-mount from host tried to link `libcuda.so.1` but failed (non-zero exit).
- Job 38380893 failed: `language_model_only=True` confirmed in logs but `Qwen3_5ForConditionalGeneration` still "failed to be inspected" — transformers 4.57.6 doesn't know Qwen3.5.
- Discovered: vLLM 0.18.0 pins `transformers<5`, but Qwen3.5 needs `transformers>=5.2.0`.
- Pinned `transformers==5.3.0` — CI failed: pip resolver conflict with vllm's `transformers<5` constraint.
- Fixed: two-stage install — vllm first, then `pip install --no-deps transformers==5.3.0`. CI passed.
- Pulled new image, streamed to Leonardo, built sandbox.
- Job 38382229 failed: Triton gcc linking against `libcuda.so.1` fails inside Singularity `--cleanenv --nv` container. The `--cleanenv` strips library paths that `--nv` sets, and `nvidia/cuda:12.4.1-runtime` doesn't have CUDA stubs.
- **Root cause of all Triton issues:** bare CUDA runtime image lacks development files needed for Triton JIT.
- Switched Dockerfile base to `vllm/vllm-openai:v0.18.0` — official vLLM image with pre-configured CUDA, Triton, gcc, and correct library paths.
- CI build triggered, waiting for completion.
- **Next:** Pull new image from CI, stream to Leonardo, rebuild sandbox, submit debug job.

### 2026-03-23
- `singularity pull` from GHCR OOM-killed on login node (twice).
- Attempted interactive Slurm job for pull — compute nodes have no internet.
- Docker image pulled on Mac, saved as `.tar.gz`, uploaded to Leonardo via scp (5.2GB).
- `singularity build` from `.tar.gz` failed with "invalid tar header".
- Gunzipped to `.tar`, `singularity build` to `.sif` still OOM-killed.
- `singularity build --sandbox` succeeded.

### 2026-03-19
- Added GitHub Actions CI workflow to build `linux/amd64` Docker image and push to GHCR.
- Switched Dockerfile base from `cuda:12.2.2-devel` to `cuda:12.4.1-runtime`.
- Pinned vllm to `>=0.8.0,<0.9.0`.
- Consolidated `leonardo.md` and `leonardo-lessons.md` into single operator reference.
- Removed obsolete `singularity/` directory.
- Rewrote `deploy.md` as clean phased guide.

### 2026-03-18
- Expanded schema with city archetypes, macro context, latent traits, micro shocks.
- Replaced cartesian-product seeds with Monte Carlo sampling.
- Updated system prompt, validation, mock backend.
- Added `--archetypes` CLI flag.
- All smoke tests pass.

### 2026-03-16
- Implemented first Qwen activity-chain generator.
- Added `activity_chain/` package with mock and vllm backends.
- Added Leonardo Slurm wrapper and workflow docs.
- Pruned legacy SAT / Modal / Vast files.
- Reduced repo to activity-chain generation only.
