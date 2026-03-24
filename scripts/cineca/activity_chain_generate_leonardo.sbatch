#!/bin/bash
#SBATCH --job-name=bonzai-qwen-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_P02_222
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Usage example:
#   sbatch \
#     --export=ALL,PROJECT_ROOT=$WORK/bonzai/activity-model,ABM_LLM_MODEL=Qwen/Qwen3.5-9B,CONTAINER_IMAGE=$WORK/containers/bonzai-qwen-generator.sif,OUTPUT_ROOT=$WORK/bonzai/activity_chain_data,TOTAL_RECORDS=1000,NUM_SHARDS=1,SHARD_ID=0 \
#     scripts/cineca/activity_chain_generate_leonardo.sbatch
#
# Notes:
# - Replace ABM_LLM_MODEL with the exact official Hugging Face model id you are serving on Leonardo.
# - For short smoke tests on Leonardo, keep --partition=boost_usr_prod and submit with --qos=boost_qos_dbg.
# - Prefer one shard per GPU job instead of multi-GPU tensor parallelism for generation.

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the repo checkout on CINECA}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to an Apptainer/Singularity image with vLLM installed}"
ABM_LLM_MODEL="${ABM_LLM_MODEL:?Set ABM_LLM_MODEL to the Qwen model id}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to the destination directory for shard JSONL files}"

TOTAL_RECORDS="${TOTAL_RECORDS:-1000}"
NUM_SHARDS="${NUM_SHARDS:?Set NUM_SHARDS explicitly}"
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-}}"
SHARD_ID="${SHARD_ID:?Set SHARD_ID explicitly or submit as a Slurm array job}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_TOKENS="${MAX_TOKENS:-2200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
DTYPE="${DTYPE:-bfloat16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
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
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_ROOT/vllm}"

export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export VLLM_CACHE_ROOT
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

unset PYTHONPATH
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

cd "$PROJECT_ROOT"

if [[ "$CONTAINER_RUNTIME" == "apptainer" ]]; then
  export APPTAINERENV_HF_HOME="$HF_HOME"
  export APPTAINERENV_HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
  export APPTAINERENV_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
  export APPTAINERENV_VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT"
  export APPTAINERENV_TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM"
  export APPTAINERENV_PYTHONUNBUFFERED="$PYTHONUNBUFFERED"
  export APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER"
  export APPTAINERENV_OMP_NUM_THREADS="$OMP_NUM_THREADS"
else
  export SINGULARITYENV_HF_HOME="$HF_HOME"
  export SINGULARITYENV_HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
  export SINGULARITYENV_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
  export SINGULARITYENV_VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT"
  export SINGULARITYENV_TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM"
  export SINGULARITYENV_PYTHONUNBUFFERED="$PYTHONUNBUFFERED"
  export SINGULARITYENV_HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER"
  export SINGULARITYENV_OMP_NUM_THREADS="$OMP_NUM_THREADS"
fi

ARGS=(
  python3
  scripts/generate_activity_chains_qwen.py
  --backend vllm
  --model "$ABM_LLM_MODEL"
  --output "$OUT_JSONL"
  --total "$TOTAL_RECORDS"
  --batch-size "$BATCH_SIZE"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --max-tokens "$MAX_TOKENS"
  --max-model-len "$MAX_MODEL_LEN"
  --dtype "$DTYPE"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --num-shards "$NUM_SHARDS"
  --shard-id "$SHARD_ID"
)

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
  --bind "$VLLM_CACHE_ROOT":"$VLLM_CACHE_ROOT" \
  "$CONTAINER_IMAGE" \
  "${ARGS[@]}"
