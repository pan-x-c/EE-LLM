import torch
import numpy as np
import torch.nn.functional as F

from megatron import get_tokenizer, get_args
from megatron.text_generation.sampling import sample
from megatron.text_generation.communication import send_token_and_probs_to_first_pipeline_stage
from megatron.core import mpu

class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    def __init__(self, max_batch_size, max_sequence_length,
                 top_k=0, top_p=0, temperature=1.0,
                 top_p_decay=0, top_p_bound=0,
                 early_exit_thres=None, use_early_exit=False,
                 print_max_prob=False,
                 exit_layers=[]):
        self.max_sequence_length = max_sequence_length
        self.max_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.batch_size_offset = 0
        self.key_value_memory_dict = {}
        self.early_exit_thres = np.log(early_exit_thres) if early_exit_thres > 0 else float('-inf')
        self.use_early_exit = use_early_exit
        self.tokenizer = get_tokenizer()
        self.use_pipeline_inference = get_args().pipeline_model_parallel_size > 1
        self.top_k = top_k
        self.top_p = top_p
        self.temperature=temperature
        self.top_p_decay = top_p_decay
        self.top_p_bound = top_p_bound
        self.print_max_probs = print_max_prob
        self.exit_layers = set(exit_layers)
        self.use_all_exit = len(exit_layers) == 0

        self.has_early_exited = False
        self.is_first_step = True
        self.prev_has_early_exited = False
        self.tokens = None
        self.probs = None

    def clear_early_exit_states(self):
        self.has_early_exited = False
        self.prev_has_early_exited = False
        self.tokens = None
        self.probs = None

    def do_early_exit(self, logits, layer_num):
        if self.has_early_exited or self.prev_has_early_exited:
            return False
        if not (self.use_all_exit or (layer_num in self.exit_layers)):
            return False
        last_token_logits = logits[:, -1, :]
        log_probs = F.log_softmax(last_token_logits, dim=1)
        max_log_prob, token_id =  torch.max(log_probs[:, :], dim=1)
        token = self.tokenizer.detokenize([int(token_id[-1])])
        if self.print_max_probs:
            print(f"layer [{layer_num}]: token [{token}], prob {float(torch.exp(max_log_prob[-1]))}")
        self.has_early_exited = max_log_prob[-1] >= self.early_exit_thres
        if self.use_pipeline_inference and self.has_early_exited:
            # send token and probs to the first stage
            tokens, probs = self.get_tokens_and_probs(last_token_logits)
            self.send_to_first_pipeline_stage(tokens, probs)
            return False
        else:
            return self.has_early_exited

    def get_tokens_and_probs(self, last_token_logits):
        tokens = sample(last_token_logits,
                            top_k=self.top_k,
                            top_p=self.top_p,
                            temperature=self.temperature,
                            vocab_size=self.tokenizer.vocab_size)
        if self.top_p > 0.0 and self.top_p_decay > 0.0:
            top_p = self.top_p * self.top_p_decay
            if self.top_p_bound > 0.0:
                top_p = max(top_p, self.top_p_bound)
        indices = torch.unsqueeze(tokens, 1)
        log_probs = F.log_softmax(last_token_logits, dim=1)
        output_log_probs = torch.gather(log_probs, 1, indices)
        return tokens, output_log_probs

    def send_to_first_pipeline_stage(self, tokens, probs):
        if mpu.is_pipeline_first_stage():
            self.tokens = tokens
            self.probs = probs
        else:
            send_token_and_probs_to_first_pipeline_stage(self, tokens, probs)

    def swap_key_value_dict(self, batch_idx):
        "swap between batches"
        if len(self.key_value_memory_dict) == 0:
            raise ValueError("should not swap when dict in empty")

        for layer_number in self.key_value_memory_dict.keys():
            inference_key_memory, inference_value_memory = self.key_value_memory_dict[layer_number]
            assert (
                len(batch_idx) == inference_key_memory.shape[1]
            )  # make sure batch size is the same
            new_inference_key_memory = inference_key_memory[:, batch_idx]
            new_inference_value_memory = inference_value_memory[:, batch_idx]
            self.key_value_memory_dict[layer_number] = (
                new_inference_key_memory,
                new_inference_value_memory,
            )
