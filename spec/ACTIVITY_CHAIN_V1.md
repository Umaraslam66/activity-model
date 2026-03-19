# Activity Chain v1

## Purpose
This repository now focuses on one problem:

`State (persona + environment) -> Action (activity chain decisions)`

The current implementation target is synthetic teacher-data generation for day-level urban activity chains.

## Core Files
- `activity_chain/schema.py`: canonical JSON schema and prompt contract
- `activity_chain/seeds.py`: deterministic persona/environment seed generation and sharding
- `activity_chain/backends.py`: `mock` and `vllm` generation backends
- `activity_chain/validation.py`: schema and plausibility validation
- `scripts/generate_activity_chains_qwen.py`: top-level CLI
- `scripts/run_activity_chain_smoke_test.py`: local CPU smoke-test entrypoint

## Record Shape
Each generated record contains:
- `schema_version`
- `scenario_id`
- `persona`
- `environment`
- `activities`
- `legs`
- `summary`

The canonical schema lives in code because the mock backend, validation path, Docker image, and Leonardo job flow all consume the same Python contract.

## Backends
### `mock`
Used for:
- local CPU smoke tests
- container smoke tests
- deterministic fixture verification

This backend does not require a local GPU or model weights.

### `vllm`
Used for:
- real Qwen generation
- Docker GPU runs
- Apptainer / Leonardo production generation

## Promotion Gates
1. Local CPU smoke test passes.
2. Docker smoke test passes.
3. Apptainer image is produced from Docker.
4. Leonardo debug shard passes.
5. Production sharded generation begins.

## Current Non-Goals
- student-model training
- legacy SAT sentiment/action head generation
- Modal deployment
- Vast.ai deployment
