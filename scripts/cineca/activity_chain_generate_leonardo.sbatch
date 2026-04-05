#!/bin/bash
#SBATCH --job-name=bonzai-gemma-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_P02_222
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Usage example:
#   sbatch \
#     --export=ALL,PROJECT_ROOT=$WORK/bonzai/activity-model,ABM_LLM_MODEL=$FAST/bonzai_cache/huggingface/google--gemma-4-31B-it,CONTAINER_IMAGE=$WORK/containers/bonzai-qwen-generator,OUTPUT_ROOT=$WORK/bonzai/activity_chain_data,TOTAL_RECORDS=1000,NUM_SHARDS=1,SHARD_ID=0 \
#     scripts/cineca/activity_chain_generate_leonardo.sbatch
#
# Notes:
# - Gemma 4 31B on Leonardo is run here with Transformers + Accelerate across 2 GPUs.
# - Replace ABM_LLM_MODEL with the local staged model path on Leonardo.
# - For short smoke tests on Leonardo, keep --partition=boost_usr_prod and submit with --qos=boost_qos_dbg.
# - Prefer one shard per 2-GPU job.

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the repo checkout on CINECA}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to an Apptainer/Singularity image with vLLM installed}"
ABM_LLM_MODEL="${ABM_LLM_MODEL:?Set ABM_LLM_MODEL to the local Gemma model path}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to the destination directory for shard JSONL files}"

TOTAL_RECORDS="${TOTAL_RECORDS:-1000}"
NUM_SHARDS="${NUM_SHARDS:?Set NUM_SHARDS explicitly}"
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-}}"
SHARD_ID="${SHARD_ID:?Set SHARD_ID explicitly or submit as a Slurm array job}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-2200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
ARCHETYPES="${ARCHETYPES:-}"

if [[ -n "${FAST:-}" ]]; then
  CACHE_ROOT="${CACHE_ROOT:-$FAST/bonzai_cache}"
else
  CACHE_ROOT="${CACHE_ROOT:-$WORK/bonzai_cache}"
fi

RUN_NAME="${RUN_NAME:-activity_chain_v1}"

mkdir -p "$CACHE_ROOT"
mkdir -p "$OUTPUT_ROOT"

HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"

mkdir -p "$HF_HOME"

export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONNOUSERSITE=1
PYTHONPATH_VALUE="${PYTHONPATH-}"
if [[ -n "$PYTHONPATH_VALUE" ]]; then
  export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH_VALUE"
else
  export PYTHONPATH="$PROJECT_ROOT"
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export ABM_BACKEND="${ABM_BACKEND:-transformers}"
export ABM_TRUST_REMOTE_CODE="${ABM_TRUST_REMOTE_CODE:-0}"
export ABM_DEVICE_MAP="${ABM_DEVICE_MAP:-$DEVICE_MAP}"
export ABM_ATTN_IMPLEMENTATION="${ABM_ATTN_IMPLEMENTATION:-$ATTN_IMPLEMENTATION}"

unset PYTHONUSERBASE

if command -v module >/dev/null 2>&1; then
  module purge
  module load hpcx-mpi/2.19
  module load cuda/12.2
fi

CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-singularity}"
if ! command -v "$CONTAINER_RUNTIME" >/dev/null 2>&1; then
  if command -v singularity >/dev/null 2>&1; then
    CONTAINER_RUNTIME="singularity"
  elif command -v apptainer >/dev/null 2>&1; then
    CONTAINER_RUNTIME="apptainer"
  else
    echo "Neither apptainer nor singularity is available in PATH" >&2
    exit 1
  fi
fi

if [[ ! -f "$CONTAINER_IMAGE" ]] && [[ ! -d "$CONTAINER_IMAGE" ]]; then
  echo "Container image not found: $CONTAINER_IMAGE" >&2
  exit 1
fi

if [[ ! -f "$PROJECT_ROOT/scripts/generate_activity_chains_qwen.py" ]]; then
  echo "Project root does not look correct: $PROJECT_ROOT" >&2
  exit 1
fi

OUT_JSONL="$OUTPUT_ROOT/${RUN_NAME}.shard${SHARD_ID}of${NUM_SHARDS}.jsonl"

echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Project root: $PROJECT_ROOT"
echo "Container: $CONTAINER_IMAGE"
echo "Model: $ABM_LLM_MODEL"
echo "Output: $OUT_JSONL"
echo "Shard: $SHARD_ID / $NUM_SHARDS"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "HF cache: $HF_HOME"
echo "HF offline: $HF_HUB_OFFLINE"
echo "backend: $ABM_BACKEND"
echo "trust_remote_code: $ABM_TRUST_REMOTE_CODE"
echo "device_map: $ABM_DEVICE_MAP"
echo "attn_implementation: $ABM_ATTN_IMPLEMENTATION"

cd "$PROJECT_ROOT"

CONTAINER_LD_LIBRARY_PATH="${CONTAINER_LD_LIBRARY_PATH:-/.singularity.d/libs:/usr/local/cuda/targets/x86_64-linux/lib/stubs:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64}"

if [[ "$CONTAINER_RUNTIME" == "apptainer" ]]; then
  export APPTAINERENV_HF_HOME="$HF_HOME"
  export APPTAINERENV_HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
  export APPTAINERENV_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
  export APPTAINERENV_TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM"
  export APPTAINERENV_PYTHONUNBUFFERED="$PYTHONUNBUFFERED"
  export APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER"
  export APPTAINERENV_HF_HUB_OFFLINE="$HF_HUB_OFFLINE"
  export APPTAINERENV_TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE"
  export APPTAINERENV_HF_HUB_DISABLE_TELEMETRY="$HF_HUB_DISABLE_TELEMETRY"
  export APPTAINERENV_OMP_NUM_THREADS="$OMP_NUM_THREADS"
  export APPTAINERENV_PYTHONNOUSERSITE="$PYTHONNOUSERSITE"
  export APPTAINERENV_PYTHONPATH="$PYTHONPATH"
  export APPTAINERENV_ABM_BACKEND="$ABM_BACKEND"
  export APPTAINERENV_ABM_TRUST_REMOTE_CODE="$ABM_TRUST_REMOTE_CODE"
  export APPTAINERENV_ABM_DEVICE_MAP="$ABM_DEVICE_MAP"
  export APPTAINERENV_ABM_ATTN_IMPLEMENTATION="$ABM_ATTN_IMPLEMENTATION"
  export APPTAINERENV_LD_LIBRARY_PATH="$CONTAINER_LD_LIBRARY_PATH"
else
  export SINGULARITYENV_HF_HOME="$HF_HOME"
  export SINGULARITYENV_HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
  export SINGULARITYENV_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
  export SINGULARITYENV_TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM"
  export SINGULARITYENV_PYTHONUNBUFFERED="$PYTHONUNBUFFERED"
  export SINGULARITYENV_HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER"
  export SINGULARITYENV_HF_HUB_OFFLINE="$HF_HUB_OFFLINE"
  export SINGULARITYENV_TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE"
  export SINGULARITYENV_HF_HUB_DISABLE_TELEMETRY="$HF_HUB_DISABLE_TELEMETRY"
  export SINGULARITYENV_OMP_NUM_THREADS="$OMP_NUM_THREADS"
  export SINGULARITYENV_PYTHONNOUSERSITE="$PYTHONNOUSERSITE"
  export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
  export SINGULARITYENV_ABM_BACKEND="$ABM_BACKEND"
  export SINGULARITYENV_ABM_TRUST_REMOTE_CODE="$ABM_TRUST_REMOTE_CODE"
  export SINGULARITYENV_ABM_DEVICE_MAP="$ABM_DEVICE_MAP"
  export SINGULARITYENV_ABM_ATTN_IMPLEMENTATION="$ABM_ATTN_IMPLEMENTATION"
  export SINGULARITYENV_LD_LIBRARY_PATH="$CONTAINER_LD_LIBRARY_PATH"
fi

ARGS=(
  python3
  scripts/generate_activity_chains_qwen.py
  --backend "$ABM_BACKEND"
  --model "$ABM_LLM_MODEL"
  --output "$OUT_JSONL"
  --total "$TOTAL_RECORDS"
  --batch-size "$BATCH_SIZE"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --max-tokens "$MAX_TOKENS"
  --max-model-len "$MAX_MODEL_LEN"
  --dtype "$DTYPE"
  --device-map "$ABM_DEVICE_MAP"
  --attn-implementation "$ABM_ATTN_IMPLEMENTATION"
  --num-shards "$NUM_SHARDS"
  --shard-id "$SHARD_ID"
)

if [[ "$ABM_TRUST_REMOTE_CODE" == "1" ]] || [[ "${ABM_TRUST_REMOTE_CODE,,}" == "true" ]]; then
  ARGS+=(--trust-remote-code)
fi

if [[ -n "$ARCHETYPES" ]]; then
  # Space-delimited ARCHETYPES is expanded into multiple CLI values.
  # shellcheck disable=SC2206
  ARCHETYPE_ARGS=($ARCHETYPES)
  ARGS+=(--archetypes "${ARCHETYPE_ARGS[@]}")
fi

srun --ntasks=1 "$CONTAINER_RUNTIME" exec --cleanenv --nv \
  --bind "$PROJECT_ROOT":"$PROJECT_ROOT" \
  --bind "$OUTPUT_ROOT":"$OUTPUT_ROOT" \
  --bind "$HF_HOME":"$HF_HOME" \
  "$CONTAINER_IMAGE" \
  "${ARGS[@]}"
