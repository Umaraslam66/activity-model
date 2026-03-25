sbatch \
    --qos=boost_qos_dbg \
    --export=ALL,PROJECT_ROOT=$WORK/bonzai/activity-
  model,ABM_LLM_MODEL=Qwen/Qwen3.5-9B,CONTAINER_IMAGE=$WORK/containers/
  bonzai-qwen-generator.sif,OUTPUT_ROOT=$WORK/bonzai/
  activity_chain_data,TOTAL_RECORDS=100,NUM_SHARDS=1,SHARD_ID=0 \
    $WORK/bonzai/activity-model/scripts/cineca/
  activity_chain_generate_leonardo.sbatch

  rsync -avz --exclude .git --exclude .venv --exclude .venv-codex--exclude .venv-smoke --exclude __pycache__ --exclude .pytest_cache--exclude data--exclude '*.sif' ./ ${LEONARDO_HOST}:\$WORK/bonzai/activity-model/

  scp /tmp/bonzai-qwen-generator.tar.gz ${LEONARDO_HOST}:/leonardo_work/AIFAC_P02_222/containers/

  docker save ghcr.io/umaraslam66/bonzai-qwen-generator:activity-chain-phase-a >/tmp/bonzai-qwen-generator.tar


  rm /tmp/bonzai-qwen-generator.tar.gz docker save ghcr.io/umaraslam66/bonzai-qwen-generator:activity-chain-phase-a > /tmp/bonzai-qwen-generator.tar


singularity build bonzai-qwen-generator.sif docker-archive://bonzai-qwen-generator.tar

sed -i 's/if \[\[ ! -f "$CONTAINER_IMAGE" \]\]; then/if [[ ! -f "$CONTAINER_IMAGE" ]] \&\& [[ ! -d "$CONTAINER_IMAGE" ]]; then/' scripts/cineca/activity_chain_generate_leonardo.sbatch

sbatch \ --export=ALL,PROJECT_ROOT=$WORK/bonzai/sentiment-action-transformer,ABM_LLM_MODEL=Qwen/Qwen3.5-9B,CONTAINER_IMAGE=$WORK/containers/bonzai-qwen-generator,OUTPUT_ROOT=$WORK/bonzai/activity_chain_data,TOTAL_RECORDS=100,NUM_SHARDS=1,SHARD_ID=0 \scripts/cineca/activity_chain_generate_leonardo.sbatch 



export HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen3.5-9B --local-dir $FAST/bonzai_cache/huggingface/Qwen--Qwen3.5-9B



docker save ghcr.io/umaraslam66/bonzai-qwen-generator:activity-chain-phase-a | ssh uaslam00@login.leonardo.cineca.it 'cat > /leonardo_work/AIFAC_P02_222/containers/bonzai-qwen-generator-v2.tar'

docker save ghcr.io/umaraslam66/bonzai-qwen-generator:activity-chain-phase-a | ssh uaslam00@login.leonardo.cineca.it 'cat > /leonardo_work/AIFAC_P02_222/containers/bonzai-qwen-generator.tar'


rsync -avz --exclude .git --exclude .venv --exclude .venv-codex --exclude .venv-smoke --exclude __pycache__ --exclude .pytest_cache --exclude data --exclude '*.sif' ./ uaslam00@login.leonardo.cineca.it:/leonardo_work/AIFAC_P02_222/bonzai/sentiment-action-transformer/  

squeue -u $USER
ls -la bonzai-qwen-gen-38382473.*

 tail -f bonzai-qwen-gen-38382473.out
tail -f bonzai-qwen-gen-38382473.err


rsync -avz --exclude .git --exclude .venv --exclude .venv-codex --exclude .venv-smoke --exclude __pycache__ --exclude .pytest_cache --exclude data --exclude '*.sif' ./ uaslam00@login.leonardo.cineca.it:/leonardo_work/AIFAC_P02_222/bonzai/sentiment-action-transformer/


docker save ghcr.io/umaraslam66/bonzai-qwen-generator:activity-chain-phase-a | ssh uaslam00@login.leonardo.cineca.it 'cat > /leonardo_work/AIFAC_P02_222/containers/bonzai-qwen-generator.tar'

rsync -avz --exclude .git --exclude .venv --exclude .venv-codex --exclude .venv-smoke --exclude __pycache__ --exclude .pytest_cache --exclude data --exclude '*.sif' ./ uaslam00@login.leonardo.cineca.it:/leonardo_work/AIFAC_P02_222/bonzai/sentiment-action-transformer/