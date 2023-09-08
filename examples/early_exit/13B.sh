#!/bin/bash

PROJECT_NAME=EE-LLM
GROUP_NAME=7B-EXIT-8-16-untie-300B

RUN_NAME=`date "+%m%d-%H%M"`

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4

# NCCL configuration
# export NCCL_IB_HCA=
# export NCCL_IB_TC=
# export NCCL_IB_SL=
# export NCCL_IB_GID_INDEX=
# export NCCL_SOCKET_IFNAME=
# export NCCL_DEBUG=WARN

# Checkpoint configuration
CHECKPOINT_HOME=
CHECKPOINT_PATH=$CHECKPOINT_HOME/$PROJECT_NAME/$GROUP_NAME

# data configuration
DATA_HOME=
TOKENIZER_PATH=
DATASET_ARXIV=${DATA_HOME}/redpajama-arxiv/all
DATASET_BOOKS=${DATA_HOME}/redpajama-book/all
DATASET_C4=${DATA_HOME}/redpajama-c4/all
DATASET_CC=${DATA_HOME}/redpajama-cc/all
DATASET_STACKEXCHANGE=${DATA_HOME}/redpajama-pile-stackexchange/all
DATASET_CODE=${DATA_HOME}/redpajama-stack-code/all
DATASET_WIKIPEDIA=${DATA_HOME}/redpajama-wiki/all
DATASET_PILE_EUROPARL=${DATA_HOME}/the-pile-europarl/all
DATASET_PILE_FREELAW=${DATA_HOME}/the-pile-freelaw/all
DATASET_PILE_HACKERNEWS=${DATA_HOME}/the-pile-hackernews/all
DATASET_PILE_NIH=${DATA_HOME}/the-pile-nih/all
DATASET_PILE_PHILPAPER=${DATA_HOME}/the-pile-philpaper/all
DATASET_PILE_PMA=${DATA_HOME}/the-pile-pubmed-abstract/all
DATASET_PILE_PMC=${DATA_HOME}/the-pile-pubmed-central/all
DATASET_PILE_USPTO=${DATA_HOME}/the-pile-uspto/all

DATA_PATH="\
    0.0362 ${DATASET_ARXIV} \
    0.0657 ${DATASET_BOOKS} \
    0.2264 ${DATASET_C4} \
    0.4491 ${DATASET_CC} \
    0.0246 ${DATASET_STACKEXCHANGE} \
    0.0810 ${DATASET_CODE} \
    0.0548 ${DATASET_WIKIPEDIA} \
    0.0010 ${DATASET_PILE_EUROPARL} \
    0.0162 ${DATASET_PILE_FREELAW} \
    0.0006 ${DATASET_PILE_HACKERNEWS} \
    0.0005 ${DATASET_PILE_NIH} \
    0.0006 ${DATASET_PILE_PHILPAPER} \
    0.0065 ${DATASET_PILE_PMA} \
    0.0318 ${DATASET_PILE_PMC} \
    0.0050 ${DATASET_PILE_USPTO} \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model $TOKENIZER_PATH \
    --split 990,9,1 \
"

# Distributed configuration
# MASTER_ADDR=127.0.0.1
# MASTER_PORT=5900
# RANK=0
# WORLD_SIZE=2
NPROC_PER_NODE=8

DIST_ARGS="
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    "

# Parallisim configuration
TP=2
PP=4

MICRO_BATCH=1
GLOBAL_BATCH=2048

# Train iteration
LOG_INTERVAL=2
SAVE_INTERVAL=$(( 240 * 10 )) # 10B data
TRAIN_ITER=$(( $SAVE_INTERVAL * 80)) # 800B data
EVAL_INTERVAL=$(( 240 * 5))

# GPT configuration
NLAYERS=40
HIDDEN=5120
HEADS=40
SEQ=2048

GPT_ARGS="
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads $HEADS \
    --seq-length $SEQ \
    --max-position-embeddings $SEQ \
    --sequence-parallel \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --lr 0.0003 \
    --train-iters $TRAIN_ITER \
    --lr-decay-style cosine \
    --min-lr 3.0e-5 \
    --weight-decay 1e-1 \
    --lr-warmup-iters 2000 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.01 \
    --clip-grad 1.0 \
    --bf16 \
    --disable-bias-linear \
    --use-flash-attn \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --swiglu \
    --untie-embeddings-and-output-weights \
"

# Early-exit configuration
EE_ARGS="
    --untie-exit-output-weights \
    --exit-layer-nums 11 21 \
    --exit-layer-weight 0.1 0.2 \
    --pre-exit \
"

OUTPUT_ARGS="
    --log-interval 2 \
    --log-timers-to-tracker \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters 0 \
    --wandb-project $PROJECT_NAME \
    --wandb-group $GROUP_NAME \
    --wandb-exp-name $RUN_NAME \
"

torchrun $DIST_ARGS \
    pretrain_early_exit_gpt.py \
    $GPT_ARGS \
    $EE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
