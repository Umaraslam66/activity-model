# Bonzai Activity-Chain Model

## Current Project Status
This repository is now in a **cleaned Phase 1 state**.

The branch is focused on one thing only:
- generate synthetic activity-chain data locally and on EuroHPC / CINECA

What is already implemented:
- canonical activity-chain schema and prompt contract (v1, with city archetypes + macro context)
- Monte Carlo seed generation with archetype-specific probability distributions
- city archetypes: `dense_transit_hub`, `sprawling_car_city`, `winter_cycling_city`
- macro context: `transit_density_index`, `active_mobility_index`, `spatial_sprawl_score`
- latent persona traits: `value_of_time_multiplier`, `weather_resilience`, `comfort_preference`
- micro context shocks: `dynamic_cost_sek`, `network_delay_mins`
- strict record validation with float-tolerant comparison
- `mock` backend for local CPU smoke tests (distances scaled by sprawl)
- `vllm` backend for real Qwen generation
- top-level generation CLI with `--archetypes` filter
- local smoke-test script
- Docker runtime source image (CUDA 12.4 runtime base, deps-only)
- GitHub Actions CI workflow to build linux/amd64 image and push to GHCR
- Apptainer conversion/run documentation
- Leonardo Slurm batch wrapper (QOS: `boost_usr_prod`, `ARCHETYPES` passthrough)

What is **not** implemented yet:
- student-model training pipeline
- feature flattening for supervised training
- ONNX export
- integration with the separate C++ engine repo

## Scope
This repository is the **model-side workspace** for Bonzai's activity-chain system.

The C++/CUDA routing engine, simulator runtime, and transport-network execution code live in a separate repository and are intentionally not present here.

This repo is responsible for the "brain" side:
- define the training data contract
- generate synthetic supervision
- validate generated records
- prepare for student-model training later

## Core Goal
Given a `State = Persona + Environment`, predict an `Action = Activity + Transport Mode`.

In the current Phase 1 implementation, that goal is represented as:
- generate full-day synthetic activity-chain records
- use those records later to train a smaller runtime student model

## Directory Map
### `activity_chain/`
This is the main Python package for the current system.

Contents:
- `schema.py`: canonical `activity_chain_v1` schema, enums, city archetypes, macro context, and the Qwen system prompt
- `seeds.py`: Monte Carlo persona/environment/macro seed generation with archetype-specific distributions
- `prompting.py`: user prompt and chat prompt assembly
- `validation.py`: structural and plausibility validation
- `backends.py`: generation backends
  - `MockBackend` for local CPU smoke tests
  - `VllmBackend` for real Qwen generation
- `runner.py`: shared generation loop, file writing, validation, and reject handling

This directory is the actual implementation surface of the project.

### `scripts/`
Thin entrypoints and operator-facing scripts.

Contents:
- `generate_activity_chains_qwen.py`: main CLI for synthetic generation
- `run_activity_chain_smoke_test.py`: local CPU smoke-test workflow
- `cineca/activity_chain_generate_leonardo.sbatch`: Leonardo Slurm batch wrapper

Rule:
- keep this directory thin
- business logic belongs in `activity_chain/`

### `spec/`
Human-readable documentation for the current workflow.

Contents:
- `ACTIVITY_CHAIN_V1.md`: high-level spec for the current activity-chain target
- `LEONARDO_SYNTHETIC_DATA_WORKFLOW.md`: generation workflow for local -> Docker -> Apptainer -> Leonardo
- `fixtures/activity_chain_mock_expected_first_record.json`: deterministic smoke-test fixture for the first mock record

Rule:
- this directory should explain the system
- it should not contain legacy or alternative workflows

### `.github/workflows/`
CI pipeline for building and publishing the runtime container.

Contents:
- `build-container.yml`: builds `linux/amd64` Docker image and pushes to GHCR

Purpose:
- reproducible container builds without local disk constraints
- image is pulled as `.sif` on Leonardo via `singularity pull`

### `docker/`
Source container definition for the runtime environment.

Contents:
- `Dockerfile.qwen-generator`: Docker-compatible source image (CUDA 12.4 runtime base)
- `README.md`: how to build and smoke-test the image locally, and how CI works

Purpose:
- CI builds via GitHub Actions (primary path)
- local runtime validation (optional)
- source artifact for Singularity `.sif` on Leonardo

### `apptainer/`
HPC-facing container notes.

Contents:
- `README.md`: how to convert the Docker image into Apptainer and run it on CINECA

Purpose:
- final delivery shape for EuroHPC / Leonardo

### Local root files
- `claude.md`: this file; the living project brief
- `deploy.md`: step-by-step EuroHPC Leonardo deployment guide (local → CI → Leonardo)
- `leonardo.md`: operator reference — hardware facts, lessons learned, session checklists
- `requirements_smoke.txt`: minimal local smoke-test dependencies
- `requirements_inference.txt`: real runtime dependencies for Qwen/vLLM generation
- `.dockerignore`: keeps the runtime image small and avoids copying junk into the container

### Not part of the repo logic
- `.venv/`: local virtualenv for development only; not part of the tracked project design
- `.git/`: git metadata

## Runtime Modes
### Local CPU smoke
Use the `mock` backend.

Purpose:
- test schema flow
- test file output
- test validation and reject handling
- test shard partitioning

This path does **not** require:
- local GPU
- local Qwen weights
- vLLM

### Real generation
Use the `vllm` backend with a Qwen model id.

Purpose:
- generate real synthetic activity-chain records
- run in Docker locally if GPU exists
- run in Apptainer on Leonardo

## Current Workflow
The intended workflow is now:

1. Run local CPU smoke test
2. Push to GitHub — CI builds `linux/amd64` Docker image and pushes to GHCR
3. On Leonardo: `singularity pull` the image from GHCR as a `.sif`
4. Sanity-check the `.sif` (python version, vllm import)
5. Run container smoke test on Leonardo (CPU, mock backend)
6. Run short Leonardo debug shard (100 records, 1 GPU)
7. Run sharded production generation (50k-100k records)
8. Only then begin student-model work

Platform note:
- Leonardo Booster nodes are x86_64 (Intel Ice Lake + A100).
- The Docker image is built as `linux/amd64` in CI (GitHub Actions), not locally.
- No local Apptainer or Docker build is needed on the Mac.

## State and Action Context
### State
Each scenario seed contains:

**Top-level:**
- `city_archetype`: one of `dense_transit_hub`, `sprawling_car_city`, `winter_cycling_city`

**Macro context (archetype-specific Gaussians):**
- `transit_density_index` (0-1)
- `active_mobility_index` (0-1)
- `spatial_sprawl_score` (0-1)

**Persona (discrete draws + Gaussian latent traits):**
- age, income_bracket, employment_status, household_structure, has_children
- car_ownership, bike_ownership, transit_pass, work_schedule
- `value_of_time_multiplier` (0.5-2.0)
- `weather_resilience` (0-1)
- `comfort_preference` (0-1)

**Environment (archetype-weighted categoricals + Gaussian shocks):**
- day_type, weather, transit_disruption, road_congestion
- parking_cost_level (archetype-weighted)
- toll_level (archetype-weighted)
- special_event
- `dynamic_cost_sek` (0-500, archetype-specific distribution)
- `network_delay_mins` (0-60, archetype-specific distribution)

### Action
Current generated output contains:
- ordered activities
- ordered travel legs
- mode choice
- timing
- summary fields

This is still teacher-data generation, not yet the final student-model interface.

## Infrastructure Context
Training and large-scale generation are expected to run on EuroHPC resources exposed through CINECA.

Current assumptions:
- Teacher model: `Qwen/Qwen3.5-9B` via vLLM on A100 64GB
- GPU generation runs on Leonardo Booster partition (`boost_usr_prod`)
- outputs should live in `$WORK`
- cache should use `$FAST` if available, otherwise `$WORK`
- model weights should not be baked into the runtime image
- one GPU job should handle one shard

Operational rule:
- keep the local workflow light and smoke-testable
- keep the HPC workflow reproducible and deterministic

## Next Work Items
The next meaningful steps after the current branch state are:
1. ~~run the smoke workflow end-to-end locally~~ DONE
2. ~~set up GitHub Actions CI to build linux/amd64 image and push to GHCR~~ DONE
3. push branch to trigger CI build
4. on Leonardo: `singularity pull` the GHCR image
5. sanity-check `.sif` (python version, `import vllm`)
6. run container smoke test on Leonardo (mock backend)
7. rsync latest repo code to Leonardo
8. run one Leonardo debug shard (100 records, 1 GPU)
9. inspect valid/reject ratio and archetype distribution
10. run sharded production generation (50k-100k records)
11. only then begin student-model dataset flattening

## Progress Log
### 2026-03-19
- Added GitHub Actions CI workflow to build `linux/amd64` Docker image and push to GHCR.
- Switched Dockerfile base from `cuda:12.2.2-devel` to `cuda:12.4.1-runtime` (smaller, vllm uses prebuilt wheels).
- Pinned vllm to `>=0.8.0,<0.9.0` for stability.
- Consolidated `leonardo.md` and `leonardo-lessons.md` into a single operator reference.
- Removed obsolete `singularity/` directory (cannot use `--fakeroot` on Leonardo).
- Removed `instructions.md` (superseded by `claude.md`).
- Rewrote `deploy.md` as a clean phased guide: Local → CI → Leonardo → GPU.
- Updated `apptainer/README.md`, `docker/README.md`, and `spec/LEONARDO_SYNTHETIC_DATA_WORKFLOW.md` to reflect CI-first flow.
- Updated `.dockerignore` to exclude dev artifacts.

### 2026-03-18
- Expanded schema with city archetypes (`dense_transit_hub`, `sprawling_car_city`, `winter_cycling_city`).
- Added `macro_context` block (transit density, active mobility, spatial sprawl).
- Added latent persona traits (value_of_time_multiplier, weather_resilience, comfort_preference).
- Added micro context shocks (dynamic_cost_sek, network_delay_mins).
- Replaced cartesian-product seed generation with Monte Carlo sampling using archetype-specific distributions.
- Updated system prompt to explain all new fields for behavioral plausibility.
- Updated validation to use `math.isclose()` for float fields.
- Mock backend distances now scale by `spatial_sprawl_score`.
- Added `--archetypes` CLI flag and `RunConfig.archetypes`.
- Fixed Leonardo sbatch QOS to `boost_usr_prod`.
- Added `ARCHETYPES` env var passthrough to sbatch.
- Regenerated deterministic mock fixture.
- Deleted legacy `Kopia av groq_50k.fixed.jsonl` (57MB).
- All smoke tests pass.

### 2026-03-16
- Clarified that the routing engine is in a separate repository and is intentionally absent here.
- Locked the repo goal to `State -> Action` for Bonzai activity-chain modeling.
- Implemented the first Qwen activity-chain generator.
- Added deterministic persona/environment seed generation and plausibility validation.
- Added a Leonardo synthetic-data workflow note and Slurm wrapper.
- Refactored the generator into the `activity_chain/` package.
- Added `mock` and `vllm` backends.
- Added deterministic smoke-test fixture coverage.
- Added local Docker and Apptainer workflow docs.
- Pruned legacy SAT / Modal / Vast files from the branch.
- Reduced the repository to the activity-chain generation workflow only.
- Updated this file to reflect the cleaned repo structure and current status.

## Working Rule
When updating this file in future work:
- keep it aligned with the actual directory tree
- explain what remains in the repo and why
- record any change to the model I/O contract
- record any change to the local -> Docker -> Apptainer -> Leonardo workflow
