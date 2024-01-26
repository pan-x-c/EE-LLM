#!/bin/bash

PROJECT_NAME=EE-TUNE
GROUP_NAME=llama-2-70B-chat-1-EXIT-pt

CURRENT_TIME=`date "+%m%d-%H%M"`

RUN_NAME=${CURRENT_TIME}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4

# Checkpoint configuration
MODEL_HOME=
LOAD_PATH=${MODEL_HOME}/checkpoints/MET-EXP/llama2-70b-chat-8-exit # your checkpoint path
CHECKPOINT_PATH=${MODEL_HOME}/checkpoints/$PROJECT_NAME/$GROUP_NAME
TOKENIZER_PATH=${MODEL_HOME}/tokenizer/tokenizer.model

# Data configuration
DATA_HOME=
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

NLAYERS=80
HIDDEN=8192
HEADS=64
SEQ=2048
FFN_SIZE=28672

TP=1
PP=4

MICRO_BATCH=4
GLOBAL_BATCH=16


MASTER_ADDR=127.0.0.1
MASTER_PORT=5900
WORLD_SIZE=1
RANK=0
NPROC_PER_NODE=4

TRAIN_ITER=40000
EVAL_INTERVAL=40000
SAVE_INTERVAL=20000

DIST_ARGS="
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    "

GPT_ARGS="
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads $HEADS \
    --seq-length $SEQ \
    --max-position-embeddings $SEQ \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --lr 0.0001 \
    --train-iters $TRAIN_ITER \
    --sequence-parallel \
    --min-lr 1.0e-5 \
    --lr-warmup-fraction .01 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --clip-grad 1.0 \
    --bf16 \
    --disable-bias-linear \
    --use-flash-attn \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --swiglu \
    --group-query-attention \
    --num-query-groups 8 \
    --exit-layer-nums 20 \
    --use-exit-norm \
    --use-exit-mlp \
    --untie-embeddings-and-output-weights \
    --untie-exit-output-weights \
    --padded-vocab-size 32000 \
    --ffn-hidden-size $FFN_SIZE \
    --finetune \
    --tune-exit-pipeline-parallel-size 1 \
    --tune-exit \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model $TOKENIZER_PATH \
    --split 990,9,1 \
"

OUTPUT_ARGS="
    --log-interval 10 \
    --log-timers-to-tracker \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters 1 \
    --wandb-project $PROJECT_NAME \
    --wandb-group $GROUP_NAME \
    --wandb-exp-name $RUN_NAME \
"

CUR_DIR=$(cd $(dirname "$0") && pwd)
MEGATRON_ROOT_PATH=$(cd "$CUR_DIR/../.." && pwd)
cd $MEGATRON_ROOT_PATH

torchrun $DIST_ARGS \
    pretrain_early_exit_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --load $LOAD_PATH \
    --save $CHECKPOINT_PATH
