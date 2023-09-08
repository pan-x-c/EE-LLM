"""Early-exit GPT model."""

import torch
import torch.nn.functional as F

from megatron import get_args
from megatron.core import tensor_parallel, mpu
from functools import partial
from .module import MegatronModule

from .enums import AttnMaskType
from .language_model import parallel_lm_logits
from .language_model import get_language_model


def post_language_model_processing(lm_output, labels, logit_weights,
                                   parallel_output,
                                   fp16_lm_cross_entropy,
                                   temperature=1.0,
                                   log_dict=None,
                                   log_key=None):

    # Output. Format [s b h]
    output = parallel_lm_logits(
        lm_output,
        logit_weights,
        parallel_output)

    if labels is None:
        # [s b h] => [b s h]
        return output.transpose(0,1).contiguous()
    else:
        # [b s] => [s b]
        labels = labels.transpose(0,1).contiguous()
        if temperature != 1.0:
            output.div_(temperature)
        if fp16_lm_cross_entropy:
            assert output.dtype == torch.half
            loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels)
        else:
            loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)
        
        # [s b] => [b, s]
        loss = loss.transpose(0,1).contiguous()
        return loss


def early_exit_processing(lm_output, labels, logit_weights,
                          parallel_output,
                          fp16_lm_cross_entropy,
                          temperature=1.0,
                          log_dict=None,
                          log_key=None):
    output = parallel_lm_logits(
        lm_output,
        logit_weights,
        parallel_output)

    if labels is None:
        # [s b h] => [b s h]
        return output.transpose(0,1).contiguous()
    else:
        # [b s] => [s b]
        labels = labels.transpose(0,1).contiguous()

        if temperature != 1.0:
            output.div_(temperature)

        with torch.no_grad():
            max_log_probs, max_idx = torch.max(F.log_softmax(output, dim=2), dim=2)
            dynamic_loss_weights = torch.exp(max_log_probs)
            if log_dict:
                log_dict[log_key] = dynamic_loss_weights.mean()

        if fp16_lm_cross_entropy:
            assert output.dtype == torch.half
            loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels)
        else:
            loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)

        loss.multiply_(dynamic_loss_weights)

        # [s b] => [b, s]
        loss = loss.transpose(0,1).contiguous()
        return loss


class EarlyExitGPTModel(MegatronModule):
    """Early-exit GPT Language model."""

    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True):
        args = get_args()
        super().__init__(config=config, share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights)

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights

        self.language_model, self._language_model_key = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            pre_process=self.pre_process,
            post_process=self.post_process)

        self.has_early_exit = mpu.has_early_exit()
        self.use_dynamic_exit_layer_weight = args.use_dynamic_exit_layer_weight

        if not args.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()

        if self.has_early_exit:
            self.exit_layer_loss_weight = dict(filter(lambda p: p[0] in mpu.get_early_exit_layer_nums(), \
                        zip(args.exit_layer_nums, args.exit_layer_weight)))
            self.exit_layer_temperature = dict(filter(lambda p: p[0] in mpu.get_early_exit_layer_nums(), \
                        zip(args.exit_layer_nums, args.exit_layer_temperature)))
            self.language_model.initialize_exit_output_weights(config, self.shared_embedding_or_output_weight() \
                        if not args.untie_embeddings_and_output_weights else None)

        if self.post_process:
            self.output_weight = self.get_output_weight()

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def get_output_weight(self):
        if self.untie_embeddings_and_output_weights:
            return self.language_model.output_layer.weight
        elif self.pre_process:
            return self.language_model.embedding.word_embeddings.weight
        else:
            return self.word_embeddings.weight

    def forward(self, input_ids, position_ids, attention_mask,
                retriever_input_ids=None,
                retriever_position_ids=None,
                retriever_attn_mask=None,
                labels=None, tokentype_ids=None,
                inference_params=None,
                exit_loss_func=None):

        early_exit_output = list()
        if self.has_early_exit:
            exit_process_func = partial(
                early_exit_processing if self.use_dynamic_exit_layer_weight else post_language_model_processing,
                labels=labels,
                parallel_output=self.parallel_output,
                fp16_lm_cross_entropy=self.fp16_lm_cross_entropy
            )

            lm_output, early_exit_output = self.language_model(
                input_ids,
                position_ids,
                attention_mask,
                retriever_input_ids=retriever_input_ids,
                retriever_position_ids=retriever_position_ids,
                retriever_attn_mask=retriever_attn_mask,
                inference_params=inference_params,
                exit_process_func=exit_process_func,
                exit_loss_func=exit_loss_func)
        else:
            lm_output = self.language_model(
                input_ids,
                position_ids,
                attention_mask,
                retriever_input_ids=retriever_input_ids,
                retriever_position_ids=retriever_position_ids,
                retriever_attn_mask=retriever_attn_mask,
                inference_params=inference_params)

        if inference_params is not None and inference_params.has_early_exited:
            return lm_output
        elif self.post_process:
            lm_output = post_language_model_processing(
                lm_output, labels,
                self.output_weight,
                self.parallel_output,
                self.fp16_lm_cross_entropy)
        if self.has_early_exit and inference_params is None:
            return lm_output, early_exit_output
        else:
            return lm_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars)
        # Save word_embeddings.
        if mpu.is_output_embedding_pipeline_stage() and not self.pre_process and not self.untie_embeddings_and_output_weights:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if mpu.is_output_embedding_pipeline_stage() and not self.pre_process and not self.untie_embeddings_and_output_weights:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)
