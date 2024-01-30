#!/bin/bash

LOAD_DIR= # path to the converted llama checkpoint in megatron format
SAVE_DIR= # path to save the converted EE LLM checkpoint

LOAD_ITER=1
CUR_DIR=$(cd $(dirname "$0") && pwd)
MEGATRON_ROOT_PATH=$(cd "$CUR_DIR/../../.." && pwd)

# For llama2 13B model (40 layers)

## add an embedding only exit every 1/8 depth
# python ${MEGATRON_ROOT_PATH}/tools/checkpoint/checkpoint_converter.py \
#     --load-dir $LOAD_DIR \
#     --save-dir $SAVE_DIR \
#     --load-iteration $LOAD_ITER \
#     --conversion-type add-exit \
#     --add-exit-layer-nums 5 10 15 20 25 30 35 40 \
#     --megatron-path $MEGATRON_ROOT_PATH

## add an embedding-norm exit every 1/8 depth
# python ${MEGATRON_ROOT_PATH}/tools/checkpoint/checkpoint_converter.py \
#     --load-dir $LOAD_DIR \
#     --save-dir $SAVE_DIR \
#     --load-iteration $LOAD_ITER \
#     --conversion-type add-exit \
#     --add-exit-layer-nums 5 10 15 20 25 30 35 40 \
#     --megatron-path $MEGATRON_ROOT_PATH

## add an embedding-norm-mlp exit every 1/8 depth
python ${MEGATRON_ROOT_PATH}/tools/checkpoint/checkpoint_converter.py \
    --load-dir $LOAD_DIR \
    --save-dir $SAVE_DIR \
    --load-iteration $LOAD_ITER \
    --use-exit-norm \
    --use-exit-mlp \
    --conversion-type add-exit \
    --add-exit-layer-nums 5 10 15 20 25 30 35 40 \
    --megatron-path $MEGATRON_ROOT_PATH

## add an embedding-norm-layer exit every 1/8 depth
# python ${MEGATRON_ROOT_PATH}/tools/checkpoint/checkpoint_converter.py \
#     --load-dir $LOAD_DIR \
#     --save-dir $SAVE_DIR \
#     --load-iteration $LOAD_ITER \
#     --use-exit-norm \
#     --use-exit-block \
#     --conversion-type add-exit \
#     --add-exit-layer-nums 5 10 15 20 25 30 35 40 \
#     --megatron-path $MEGATRON_ROOT_PATH

## add an embedding-norm-mlp exit at 1/4 depth
# python ${MEGATRON_ROOT_PATH}/tools/checkpoint/checkpoint_converter.py \
#     --load-dir $LOAD_DIR \
#     --save-dir $SAVE_DIR \
#     --load-iteration $LOAD_ITER \
#     --use-exit-norm \
#     --use-exit-mlp \
#     --conversion-type add-exit \
#     --add-exit-layer-nums 10 \
#     --megatron-path $MEGATRON_ROOT_PATH

## add an random init embedding-norm-mlp exit at 1/4 depth
# python ${MEGATRON_ROOT_PATH}/tools/checkpoint/checkpoint_converter.py \
#     --load-dir $LOAD_DIR \
#     --save-dir $SAVE_DIR \
#     --load-iteration $LOAD_ITER \
#     --use-exit-norm \
#     --use-exit-mlp \
#     --random-init \
#     --conversion-type add-exit \
#     --add-exit-layer-nums 10 \
#     --megatron-path $MEGATRON_ROOT_PATH

# For llama2 70B model (80 layers)

## add an embedding-norm-mlp exit every 1/8 depth
# python ${MEGATRON_ROOT_PATH}/tools/checkpoint/checkpoint_converter.py \
#     --load-dir $LOAD_DIR \
#     --save-dir $SAVE_DIR \
#     --load-iteration $LOAD_ITER \
#     --use-exit-norm \
#     --use-exit-mlp \
#     --conversion-type add-exit \
#     --add-exit-layer-nums 10 20 30 40 50 60 70 80 \
#     --megatron-path $MEGATRON_ROOT_PATH

# For llama2 7B model (32 layers)

## add an embedding-norm-mlp exit every 1/8 depth
# python ${MEGATRON_ROOT_PATH}/tools/checkpoint/checkpoint_converter.py \
#     --load-dir $LOAD_DIR \
#     --save-dir $SAVE_DIR \
#     --load-iteration $LOAD_ITER \
#     --use-exit-norm \
#     --use-exit-mlp \
#     --conversion-type add-exit \
#     --add-exit-layer-nums 4 8 12 16 20 24 28 32 \
#     --megatron-path $MEGATRON_ROOT_PATH