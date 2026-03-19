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
#     --export=ALL,PROJECT_ROOT=$WORK/bonzai/sentiment-action-transformer,ABM_LLM_MODEL=Qwen/Qwen3.5-9B,CONTAINER_IMAGE=$WORK/containers/bonzai-qwen-generator.sif,OUTPUT_ROOT=$WORK/bonzai/activity_chain_data,TOTAL_RECORDS=1000,NUM_SHARDS=1,SHARD_ID=0 \
#     scripts/cineca/activity_chain_generate_leonardo.sbatch
#
# Notes:
# - Replace ABM_LLM_MODEL with the exact local Qwen 9B model identifier you are serving on Leonardo.
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

CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-apptainer}"
if ! command -v "$CONTAINER_RUNTIME" >/dev/null 2>&1; then
  if command -v singularity >/dev/null 2>&1; then
    CONTAINER_RUNTIME="singularity"
  else
    echo "Neither apptainer nor singularity is available in PATH" >&2
    exit 1
  fi
fi

OUT_JSONL="$OUTPUT_ROOT/${RUN_NAME}.shard${SHARD_ID}of${NUM_SHARDS}.jsonl"

echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Project root: $PROJECT_ROOT"
echo "Container: $CONTAINER_IMAGE"
echo "Model: $ABM_LLM_MODEL"
echo "Output: $OUT_JSONL"
echo "Shard: $SHARD_ID / $NUM_SHARDS"

cd "$PROJECT_ROOT"

"$CONTAINER_RUNTIME" exec --nv \
  --bind "$PROJECT_ROOT":"$PROJECT_ROOT" \
  --bind "$OUTPUT_ROOT":"$OUTPUT_ROOT" \
  --bind "$HF_HOME":"$HF_HOME" \
  --bind "$VLLM_CACHE_ROOT":"$VLLM_CACHE_ROOT" \
  "$CONTAINER_IMAGE" \
  python scripts/generate_activity_chains_qwen.py \
    --backend vllm \
    --model "$ABM_LLM_MODEL" \
    --output "$OUT_JSONL" \
    --total "$TOTAL_RECORDS" \
    --batch-size "$BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --max-tokens "$MAX_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --num-shards "$NUM_SHARDS" \
    --shard-id "$SHARD_ID" \
    ${ARCHETYPES:+--archetypes $ARCHETYPES}
