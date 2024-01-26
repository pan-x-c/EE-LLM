#!/bin/bash

LOAD_DIR= # path to the llama2 huggingface checkpoint dir
SAVE_DIR= # path to save the converted megatron checkpoint
TP=1  # target tensor parallel size
PP=4  # target pipeline parallel size

TOKENIZER_PATH= ${LOAD_DIR}/tokenizer.model

CUR_DIR=$(cd $(dirname "$0") && pwd)
MEGATRON_ROOT_PATH=$(cd "$CUR_DIR/../../.." && pwd)

python $MEGATRON_ROOT_PATH/tools/checkpoint/util.py \
    --model-type EarlyExitGPT \
    --load-dir $LOAD_DIR \
    --save-dir $SAVE_DIR \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size $TP \
    --target-pipeline-parallel-size $PP \
    --megatron-path $MEGATRON_ROOT_PATH \
    --tokenizer-model $TOKENIZER_PATH