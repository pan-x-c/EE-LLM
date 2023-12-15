# README

[EE-LLM](https://arxiv.org/abs/2312.04916) is a framework for large-scale training and inference of early-exit (EE) large language models (LLMs), which is built upon [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and currently under active development.

![](images/ee_architecture.png)

## Installation

The installation of EE-LLM is the same as Megatron-LM.
We recommand using the 22.12 version of [NGC's PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) (nvcr.io/nvidia/pytorch:22.12-py3), which is also the development environment of EE-LLM.

For more details about the installation of Megatron-LM, please refer to Megatron-LM's [README](README_Megatron_LM.md).


## Training

Below are several example training scripts used in our paper.


```
# train 1.3B model
./examples/early_exit/1-3B.sh

# train 7B model
./examples/early_exit/7B.sh

# train 13B model 
./example/early_exit/13B.sh

# train 30B model
./example/early_exit/30B.sh
```


The training data used in these scripts can be found in [Data-Juicer](https://github.com/alibaba/data-juicer/blob/main/configs/data_juicer_recipes/README.md). 
You can modify the `DATA_PATH` environment variable in the scripts to use your own dataset.
Note that Megatron-LM can only recognize preprocessed binary data; 
for more details about Megatron-LM's data preprocessing, please refer to [Data Preprocessing](README_Megatron_LM.md)

> Running the training scripts requires 16 Nvidia A100-80G GPUs or higher hardware specifications. To run them with fewer GPUs, please set the parallelism degrees therein to smaller values.


Below are the new configurations of EE-LLM compared to Megatron-LM. You can customize your own early-exit LLM by modifying these configurations.

### Configurations for model architectures

- `--exit-layer-nums`: indices of the Transformer layers converted to early-exit Transformer layers, starting from 1.
    > For example, `--exit-layer-nums 6 12` will add early exits to the 6th and 12th Transformer layers.

- `--pre-exit`: If set, the early-exit modules will be placed before the backbone of the Transformer layer, otherwise they will be placed after the backbone by default.
    > For example, the overall model architectures represented by `--exit-layer-nums 6 12` and `--exit-layer-nums 7 13 --pre-exit` are the same.

- `--untie-exit-output-weights`: If set, each early exit uses a different output word embedding, otherwise all early exits share the same output word embedding.

- `--use-exit-norm`: If set, add a Norm layer before the early-exit output word embedding.

- `--use-exit-mlp`: If set, add a MLP layer before the early-exit output word embedding.

- `--use-exit-block`: If set, add a complete Transformer layer before the early-exit output word embedding.

### Configurations for training

- `--exit-layer-weight`: The targeted loss weights of early exits. Must correspond to `--exit-layer-nums` one-to-one. Default to 1.0.

- `--exit-layer-weight-init`: The initial loss weights of early exits, which can be lower or higher than `--exit-layer-weight`.

- `--exit-layer-weight-warmup-iters`: The number of warm-up/cool-down iterations for early-exit loss weights (from `weight-init` to `weight`), default to 0.

- `--exit-layer-weight-warmup-style`: The increment function of early-exit loss weights, default to linear.

- `--fill-explicit-bubbles`: Enable filling explicit bubbles of the 1F1B pipeline schedule with additional microbatches. [Experimental]

- `--num-fill-warmup-microbatches`: The number of microbatches to be inserted during the warm-up phase of the 1F1B schedule. [Experimental]

- `--num-fill-cooldown-microbatches`: The number of microbatches to be inserted during the cool-down phase of the 1F1B schedule. [Experimental]

- `--backward-forward-ratio`: An estimate of the ratio of time consumption between backward and forward computation during training, used to automatically calculate the optimal number of inserted microbatches. Default to 2.0. [Experimental]

## Inference

We provided an text generation server for inference of early-exit LLMs.
To start a server, you can use the following script.
Before running, please set `CHECKPOINT_PATH` to the root folder path of the checkpoint, and set `TP` and `PP` appropriately according to the parallelism of the checkpoint.

```
./example/early_exit/ee_inference_server.sh
```

After the server is started, you can use `tools/request_client.py` to send requests to the server.
Below are some parameters for early-exit LLM inference, which can be found in `tools/request_client.py`.

- `use_early_exit`: The early-exit feature is only enabled when this option is set, otherwise the model behaves exactly like a standard model without early exits.

- `early_exit_thres`: The confidence threshold used to determine whether to execute early exiting, ranging from 0.0 to 1.0.

- `print_max_prob`: If set, the inference server will print the token with the highest confidence and the confidence values at all exits.


## BibTeX

```
@misc{chen2023eellm,
    title={EE-LLM: Large-Scale Training and Inference of Early-Exit Large Language Models with 3D Parallelism}, 
    author={Yanxi Chen and Xuchen Pan and Yaliang Li and Bolin Ding and Jingren Zhou},
    year={2023},
    eprint={2312.04916},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```


