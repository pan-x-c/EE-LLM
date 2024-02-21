import json
import os
import sys
import torch
import argparse
import math
from collections import OrderedDict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', type=str)
    parser.add_argument('--load-iteration', type=int)
    parser.add_argument('--save-dir', type=str)
    parser.add_argument('--conversion-type', choices=['exit-position', 'add-exit'], default='add-exit')
    parser.add_argument('--target-exit-position', choices=['pre', 'post'], default='post')
    parser.add_argument('--add-exit-layer-nums', type=int, nargs='+', default=[])
    parser.add_argument('--use-exit-mlp', action='store_true')
    parser.add_argument('--use-exit-block', action='store_true')
    parser.add_argument('--use-exit-norm', action='store_true')
    parser.add_argument('--random-init', action='store_true')
    parser.add_argument('--init-method-std', type=float, default=0.02)
    parser.add_argument('--megatron-path', type=str, default=None)
    return parser.parse_args()

def load_checkpoint_args(checkpoint_root_path):
    if os.path.exists(os.path.join(checkpoint_root_path, 'mp_rank_00')):
        checkpoint_rank_0_dir = 'mp_rank_00'
    elif os.path.exists(os.path.join(checkpoint_root_path, 'mp_rank_00_000')):
        checkpoint_rank_0_dir = 'mp_rank_00_000'
    else:
        raise FileNotFoundError(f'Checkpoint file {checkpoint_root_path} not found')
    checkpoint_path = os.path.join(checkpoint_root_path, checkpoint_rank_0_dir, 'model_optim_rng.pt')
    print(f"Loading args from {checkpoint_root_path}")
    model = torch.load(checkpoint_path)
    return model['args']

# Init method from megatron-lm
def init_method_normal(sigma):

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_

def scaled_init_method_normal(sigma, num_layers):
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def change_exit_position(args, checkpoint_load_dir, checkpoint_save_dir):
    checkpoint_args = load_checkpoint_args(checkpoint_load_dir)
    cur_exit_position = 'pre' if checkpoint_args.pre_exit else 'post'
    if cur_exit_position == args.target_exit_position:
        print("No need to convert")
        return
    pipeline_parallel_size = checkpoint_args.pipeline_model_parallel_size
    tensor_parallel_size = checkpoint_args.tensor_model_parallel_size
    exit_layer_nums = checkpoint_args.exit_layer_nums
    if args.target_exit_position == 'pre':
        exit_layer_nums = [layer_num + 1 for layer_num in exit_layer_nums]
    else:
        exit_layer_nums = [layer_num - 1 for layer_num in exit_layer_nums]
    use_pipeline_parallel = pipeline_parallel_size > 1
    for tensor_rank in range(tensor_parallel_size):
        checkpoint_dicts = {}
        exit_output_weights = []
        exit_output_weight_offset = 0
        # load all pipeline ranks
        for pipeline_rank in range(pipeline_parallel_size):
            if not use_pipeline_parallel:
                checkpoint_name = os.path.join(checkpoint_load_dir, f'mp_rank_{tensor_rank:02d}', 'model_optim_rng.pt')
            else:
                checkpoint_name = os.path.join(checkpoint_load_dir, f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}', 'model_optim_rng.pt')
            print(f'Loading checkpoint [pp:{pipeline_rank}, tp:{tensor_rank}] from {checkpoint_name} ...')
            state_dict = torch.load(checkpoint_name, map_location='cpu')
            checkpoint_dicts[pipeline_rank] = state_dict
            # convert args
            state_dict['args'].exit_layer_nums = exit_layer_nums
            state_dict['args'].pre_exit = (args.target_exit_position == 'pre')
            # get exit output weight
            if checkpoint_args.untie_exit_output_weights and use_pipeline_parallel:
                if 'exit_output_layer' in state_dict['model']['language_model']:
                    exit_weight_num = len(state_dict['model']['language_model']['exit_output_layer'])
                    for i in range(exit_weight_num):
                        exit_output_weights.append(state_dict['model']['language_model']['exit_output_layer'].pop(f'{i}.weight'))
        # convert output weight position
        if checkpoint_args.untie_exit_output_weights and use_pipeline_parallel:
            layer_per_stage = checkpoint_args.num_layers / pipeline_parallel_size
            for pipeline_rank in range(pipeline_parallel_size):
                layer_nums = list(filter(lambda x: (layer_per_stage * pipeline_rank + 1) <= x <= (layer_per_stage * (pipeline_rank + 1)), exit_layer_nums))
                if len(layer_nums) > 0:
                    if 'exit_output_layer' not in checkpoint_dicts[pipeline_rank]['model']['language_model']:
                        checkpoint_dicts[pipeline_rank]['model']['language_model']['exit_output_layer'] = OrderedDict()
                    for i in range(len(layer_nums)):
                        checkpoint_dicts[pipeline_rank]['model']['language_model']['exit_output_layer'][f'{i}.weight'] = exit_output_weights[exit_output_weight_offset]
                        exit_output_weight_offset += 1
                elif 'exit_output_layer' in checkpoint_dicts[pipeline_rank]['model']['language_model']:
                    checkpoint_dicts[pipeline_rank]['model']['language_model'].pop('exit_output_layer')
        # save back
        for pipeline_rank in range(pipeline_parallel_size):
            if not use_pipeline_parallel:
                checkpoint_save_path = os.path.join(checkpoint_save_dir, f'mp_rank_{tensor_rank:02d}', 'model_optim_rng.pt')
            else:
                checkpoint_save_path = os.path.join(checkpoint_save_dir, f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}', 'model_optim_rng.pt')
            dirname = os.path.dirname(checkpoint_save_path)
            os.makedirs(dirname, exist_ok = True)
            print(f'Saving checkpoint [pp:{pipeline_rank}, tp:{tensor_rank}] to {checkpoint_save_path} ...')
            torch.save(checkpoint_dicts[pipeline_rank], checkpoint_save_path)
    print('Exit Weight Position Conversion Completed')


def add_exit(args, checkpoint_load_dir, checkpoint_save_dir):
    if len(args.add_exit_layer_nums) == 0:
        print("No exit layer to add")
        return
    checkpoint_args = load_checkpoint_args(checkpoint_load_dir)
    use_pre_exit = False

    if not hasattr(checkpoint_args, "exit_layer_nums"):
        checkpoint_args.exit_layer_nums = []
    if not hasattr(checkpoint_args, "pre_exit"):
        checkpoint_args.pre_exit = False
    
    if len(checkpoint_args.exit_layer_nums) == 0:
        if args.target_exit_position == 'pre':
            use_pre_exit = True
    else:
        if checkpoint_args.pre_exit == (args.target_exit_position == 'pre'):
            print("Can't add exit layers and change exit position at the same time")
            return
        use_pre_exit = checkpoint_args.pre_exit
    target_exit_layer_nums = sorted(list(set(checkpoint_args.exit_layer_nums + args.add_exit_layer_nums)))
    tensor_parallel_size = checkpoint_args.tensor_model_parallel_size
    pipeline_parallel_size = checkpoint_args.pipeline_model_parallel_size
    use_pipeline_parallel = pipeline_parallel_size > 1
    layer_per_stage = checkpoint_args.num_layers / pipeline_parallel_size

    for tensor_rank in range(tensor_parallel_size):
        checkpoint_dicts = {}
        output_weight = None
        final_norm_weight = None
        final_norm_bias = None
        # load all pipeline ranks
        for pipeline_rank in range(pipeline_parallel_size):
            if not use_pipeline_parallel:
                checkpoint_name = os.path.join(checkpoint_load_dir, f'mp_rank_{tensor_rank:02d}', 'model_optim_rng.pt')
            else:
                checkpoint_name = os.path.join(checkpoint_load_dir, f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}', 'model_optim_rng.pt')
            print(f'Loading checkpoint [pp:{pipeline_rank}, tp:{tensor_rank}] from {checkpoint_name} ...')
            layer_num_offset = layer_per_stage * pipeline_rank + 1
            exit_layer_nums = list(filter(lambda x: (layer_per_stage * pipeline_rank + 1) <= x <= (layer_per_stage * (pipeline_rank + 1)), target_exit_layer_nums))
            state_dict = torch.load(checkpoint_name)
            checkpoint_dicts[pipeline_rank] = state_dict
            # convert args
            state_dict['args'].exit_layer_nums = target_exit_layer_nums
            state_dict['args'].pre_exit = use_pre_exit
            state_dict['args'].untie_exit_output_weights = True

            # get ouptut weight
            if checkpoint_args.untie_embeddings_and_output_weights:
                if pipeline_rank == pipeline_parallel_size - 1:
                    output_weight = state_dict['model']['language_model']['output_layer']['weight']
            else:
                if pipeline_rank == 0:
                    output_weight = state_dict['model']['language_model']['embedding']['word_embeddings']['weight']

            # convert to exit mlp
            if args.use_exit_mlp and (not hasattr(state_dict['args'], 'use_exit_mlp') or not state_dict['args'].use_exit_mlp):
                state_dict['args'].use_exit_mlp = args.use_exit_mlp
                for layer_num in exit_layer_nums:
                    if args.random_init:
                        init_method = init_method_normal(args.init_method_std)
                        output_layer_init_method = scaled_init_method_normal(args.init_method_std, layer_num)
                    layer_id = int(layer_num - layer_num_offset)
                    state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.trunk.dense_h_to_4h.weight'] = \
                            state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.dense_h_to_4h.weight']
                    state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.trunk.dense_4h_to_h.weight'] = \
                            state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.dense_4h_to_h.weight']
                    if args.random_init:
                        state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.branch.dense_h_to_4h.weight'] = \
                                init_method(torch.empty(state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.dense_h_to_4h.weight'].shape))
                        state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.branch.dense_4h_to_h.weight'] = \
                                output_layer_init_method(torch.empty(state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.dense_4h_to_h.weight'].shape))
                    else:
                        state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.branch.dense_h_to_4h.weight'] = \
                                state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.dense_h_to_4h.weight']
                        state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.branch.dense_4h_to_h.weight'] = \
                                state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.dense_4h_to_h.weight']
                    state_dict['model']['language_model']['encoder'].pop(f'layers.{layer_id}.mlp.dense_h_to_4h.weight')
                    state_dict['model']['language_model']['encoder'].pop(f'layers.{layer_id}.mlp.dense_4h_to_h.weight')
                    if checkpoint_args.add_bias_linear:
                        state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.trunk.dense_h_to_4h.bias'] = \
                                state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.dense_h_to_4h.bias']
                        state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.trunk.dense_4h_to_h.bias'] = \
                                state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.dense_4h_to_h.bias']
                        if args.random_init:
                            state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.branch.dense_h_to_4h.bias'] = \
                                    torch.zeros(state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.dense_h_to_4h.bias'].shape)
                            state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.branch.dense_4h_to_h.bias'] = \
                                    torch.zeros(state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.dense_4h_to_h.bias'].shape)
                        else:
                            state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.branch.dense_h_to_4h.bias'] = \
                                    state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.dense_h_to_4h.bias']
                            state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.branch.dense_4h_to_h.bias'] = \
                                    state_dict['model']['language_model']['encoder'][f'layers.{layer_id}.mlp.dense_4h_to_h.bias']
                        state_dict['model']['language_model']['encoder'].pop(f'layers.{layer_id}.mlp.dense_h_to_4h.bias')
                        state_dict['model']['language_model']['encoder'].pop(f'layers.{layer_id}.mlp.dense_4h_to_h.bias')
            # convert to exit block
            if args.use_exit_block:
                state_dict['args'].use_exit_block = args.use_exit_block
                # get last layer params
                if pipeline_rank == pipeline_parallel_size - 1:
                    last_layer_id = int(layer_per_stage - 1)
                    last_layer_input_norm = state_dict['model']['language_model']['encoder'][f'layers.{last_layer_id}.input_norm.weight']
                    last_layer_atten_qkv = state_dict['model']['language_model']['encoder'][f'layers.{last_layer_id}.self_attention.query_key_value.weight']
                    last_layer_atten_dense = state_dict['model']['language_model']['encoder'][f'layers.{last_layer_id}.self_attention.dense.weight']
                    last_layer_post_norm = state_dict['model']['language_model']['encoder'][f'layers.{last_layer_id}.post_attention_norm.weight']
                    last_layer_mlp_h_to_4h = state_dict['model']['language_model']['encoder'][f'layers.{last_layer_id}.mlp.dense_h_to_4h.weight']
                    last_layer_mlp_4h_to_h = state_dict['model']['language_model']['encoder'][f'layers.{last_layer_id}.mlp.dense_4h_to_h.weight']
                    if checkpoint_args.add_bias_linear:
                        last_layer_atten_dense_bias = state_dict['model']['language_model']['encoder'][f'layers.{last_layer_id}.self_attention.dense.bias']
                        last_layer_h_to_4h_bias = state_dict['model']['language_model']['encoder'][f'layers.{last_layer_id}.mlp.dense_h_to_4h.bias']
                        last_layer_4h_to_h_bias = state_dict['model']['language_model']['encoder'][f'layers.{last_layer_id}.mlp.dense_4h_to_h.bias']
                    if checkpoint_args.normalization == 'LayerNorm':
                        last_layer_input_norm_bias = state_dict['model']['language_model']['encoder'][f'layers.{last_layer_id}.input_norm.bias']
                        last_layer_post_norm_bias = state_dict['model']['language_model']['encoder'][f'layers.{last_layer_id}.post_attention_norm.bias']
            # get final norm
            if args.use_exit_norm:
                state_dict['args'].use_exit_norm = args.use_exit_norm
                if 'final_norm.weight' in state_dict['model']['language_model']['encoder']:
                    final_norm_weight = state_dict['model']['language_model']['encoder']['final_norm.weight']
                    if checkpoint_args.normalization == 'LayerNorm':
                        final_norm_bias = state_dict['model']['language_mode']['encoder']['final_norm.bias']
            # get exit output weight
            if len(exit_layer_nums) > 0 and 'exit_output_layer' not in state_dict['model']['language_model']:
                state_dict['model']['language_model']['exit_output_layer'] = OrderedDict()

        for pipeline_rank in range(pipeline_parallel_size):
            layer_num_offset = layer_per_stage * pipeline_rank + 1
            exit_layer_nums = list(filter(lambda x: (layer_per_stage * pipeline_rank + 1) <= x <= (layer_per_stage * (pipeline_rank + 1)), target_exit_layer_nums))
            # add exit output weight and exit norm
            for i, layer_num in enumerate(exit_layer_nums):
                layer_id = int(layer_num - layer_num_offset)
                if args.random_init:
                    init_method = init_method_normal(args.init_method_std)
                    output_layer_init_method = scaled_init_method_normal(args.init_method_std, layer_num)
                    checkpoint_dicts[pipeline_rank]['model']['language_model']['exit_output_layer'][f'{i}.weight'] = init_method(torch.empty(output_weight.shape))
                else:
                    checkpoint_dicts[pipeline_rank]['model']['language_model']['exit_output_layer'][f'{i}.weight'] = output_weight
                if args.use_exit_block:
                    if args.random_init:
                        checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.input_norm.weight'] = torch.ones(last_layer_input_norm.shape)
                        checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.self_attention.query_key_value.weight'] = init_method(torch.empty(last_layer_atten_qkv.shape))
                        checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.self_attention.dense.weight'] =output_layer_init_method(torch.empty(last_layer_atten_dense.shape))
                        checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.post_attention_norm.weight'] = torch.ones(last_layer_post_norm.shape)
                        checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.mlp.dense_h_to_4h.weight'] = init_method(torch.empty(last_layer_mlp_h_to_4h.shape))
                        checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.mlp.dense_4h_to_h.weight'] = output_layer_init_method(torch.empty(last_layer_mlp_4h_to_h.shape))
                        if checkpoint_args.add_bias_linear:
                            checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.self_attention.dense.bias'] = torch.zeros(last_layer_atten_dense_bias.shape)
                            checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.mlp.dense_h_to_4h.bias'] = torch.zeros(last_layer_h_to_4h_bias.shape)
                            checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.mlp.dense_4h_to_h.bias'] = torch.zeros(last_layer_4h_to_h_bias.shape)
                        if checkpoint_args.normalization == 'LayerNorm':
                            checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.input_norm.bias'] = torch.zeros(last_layer_input_norm_bias.shape)
                            checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.post_attention_norm.bias'] = torch.zeros(last_layer_post_norm_bias.shape)
                    else:
                        checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.input_norm.weight'] = last_layer_input_norm
                        checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.self_attention.query_key_value.weight'] = last_layer_atten_qkv
                        checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.self_attention.dense.weight'] = last_layer_atten_dense
                        checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.post_attention_norm.weight'] = last_layer_post_norm
                        checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.mlp.dense_h_to_4h.weight'] = last_layer_mlp_h_to_4h
                        checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.mlp.dense_4h_to_h.weight'] = last_layer_mlp_4h_to_h
                        if checkpoint_args.add_bias_linear:
                            checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.self_attention.dense.bias'] = last_layer_atten_dense_bias
                            checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.mlp.dense_h_to_4h.bias'] = last_layer_h_to_4h_bias
                            checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.mlp.dense_4h_to_h.bias'] = last_layer_4h_to_h_bias
                        if checkpoint_args.normalization == 'LayerNorm':
                            checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.input_norm.bias'] = last_layer_input_norm_bias
                            checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_block.post_attention_norm.bias'] = last_layer_post_norm_bias
                if args.use_exit_norm:
                    if args.random_init:
                        checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_norm.weight'] = torch.ones(final_norm_weight.shape)
                    else:
                        checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_norm.weight'] = final_norm_weight
                    if final_norm_bias is not None:
                        if args.random_init:
                            checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_norm.bias'] = torch.zeros(final_norm_bias.shape)
                        else:
                            checkpoint_dicts[pipeline_rank]['model']['language_model']['encoder'][f'layers.{layer_id}.exit_norm.bias'] = final_norm_bias
            if not use_pipeline_parallel:
                checkpoint_save_path = os.path.join(checkpoint_save_dir, f'mp_rank_{tensor_rank:02d}', 'model_optim_rng.pt')
            else:
                checkpoint_save_path = os.path.join(checkpoint_save_dir, f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}', 'model_optim_rng.pt')
            dirname = os.path.dirname(checkpoint_save_path)
            os.makedirs(dirname, exist_ok = True)
            print(f'Saving checkpoint [pp:{pipeline_rank}, tp:{tensor_rank}] to {checkpoint_save_path} ...')
            torch.save(checkpoint_dicts[pipeline_rank], checkpoint_save_path)
    print('Add Exit Layers Completed')


def convert(args):
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    checkpoint_load_dir = os.path.join(args.load_dir, 'iter_{:07d}'.format(args.load_iteration))
    checkpoint_save_dir = os.path.join(args.save_dir, 'iter_{:07d}'.format(args.load_iteration))
    if args.conversion_type == 'exit-position':
        change_exit_position(args, checkpoint_load_dir, checkpoint_save_dir)
    elif args.conversion_type == 'add-exit':
        add_exit(args, checkpoint_load_dir, checkpoint_save_dir)
    with open(os.path.join(args.save_dir, 'latest_checkpointed_iteration.txt'), 'w', encoding='utf-8') as f:
        f.write(str(args.load_iteration))


if __name__ == '__main__':
    args = get_args()
    convert(args)
