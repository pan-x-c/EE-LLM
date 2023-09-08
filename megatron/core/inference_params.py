import torch
import numpy as np
import torch.nn.functional as F

class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    def __init__(self, max_batch_size, max_sequence_length, early_exit_thres=None, tokenizer=None):
        self.max_sequence_length = max_sequence_length
        self.max_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.batch_size_offset = 0
        self.key_value_memory_dict = {}
        self.early_exit_thres = np.log(early_exit_thres)
        self.has_early_exit = False
        self.is_first_step = True
        self.tokenizer = tokenizer
        self.prev_has_early_exit = False
        self.output_logits = dict()

    def early_exit(self, logits, layer_num=0):
        # to regularly recompute kv cache of the entire network
        # if self.is_first_step or logits.shape[0] >= 100:
        #     return False
        max_log_probs, token_id = torch.max(F.log_softmax(logits, dim=2), dim=2)
        token = self.tokenizer.detokenize([int(token_id[0][-1])])
        print(f"layer [{layer_num}]: token [{token}], prob {float(torch.exp(max_log_probs[0][-1]))}")
        self.has_early_exit = max_log_probs[0][-1] >= self.early_exit_thres
        return self.has_early_exit

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
