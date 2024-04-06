import sys
from typing import Optional, List, Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import LlamaForCausalLM, LlamaPreTrainedModel, LlamaConfig, AutoModel
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.models.idefics.modeling_idefics import LLAMA_INPUTS_DOCSTRING, _CONFIG_FOR_DOC
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaModel
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
import torch.distributed as dist


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    """
    prompt type1: "{}", summarize the above passage within eight words: <s1><s2><s3><s4><s5><s6><s7><s8>
    token ids: [376, ..., 9162, 19138, 675, 278, 2038, 13382, 2629, 9475, 3838, 29901, 29871, 
                32008, 32011, 32004, 32013, 32007, 32005, 32002, 32014]

    prompt type2: "{}", predict the following passage within eight words: <s9><s10><s11><s12><s13><s14><s15><s16>
    token ids: [376, ..., 8500, 278, 1494, 13382, 2629, 9475, 3838, 29901, 29871, 32000, 32009, 
                32012, 32001, 32010, 32003, 32006, 32015]
    """
    summarize_suffix_ids = [9162, 19138, 675, 278, 2038, 13382, 2629, 9475, 3838, 29901, 29871,
                            32008, 32011, 32004, 32013, 32007, 32005, 32002, 32014]
    predict_suffix_ids = [8500, 278, 1494, 13382, 2629, 9475, 3838, 29901, 29871, 32000, 32009,
                          32012, 32001, 32010, 32003, 32006, 32015]
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    expanded_mask[:, :, - len(predict_suffix_ids):, -len(summarize_suffix_ids) - len(predict_suffix_ids): - len(predict_suffix_ids)] = 0

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

class NewLlamaModel(LlamaModel):
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        # if dist.get_rank() == 0:
        #     print(combined_attention_mask[0][0][-1])

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            summarize_suffix_ids = [9162, 19138, 675, 278, 2038, 13382, 2629, 9475, 3838, 29901, 29871,
                                    32008, 32011, 32004, 32013, 32007, 32005, 32002, 32014]
            predict_suffix_ids = [8500, 278, 1494, 13382, 2629, 9475, 3838, 29901, 29871, 32000, 32009,
                                  32012, 32001, 32010, 32003, 32006, 32015]
            for i in range(len(position_ids)):
                position_ids[i][-len(predict_suffix_ids): ] = position_ids[i][
                                  -len(summarize_suffix_ids) - len(predict_suffix_ids): -len(summarize_suffix_ids)]
            """
            tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]], device='cuda:0')
            """
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
            padding_mask = None
        else:
            if 0 in attention_mask:
                padding_mask = attention_mask
            else:
                padding_mask = None

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions, padding_mask=padding_mask)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer), hidden_states, attention_mask, position_ids
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    padding_mask=padding_mask,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class PreLlamaModel(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = NewLlamaModel(config)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

        """
        prompt type1: "{}", summarize the above passage within eight words: <s1><s2><s3><s4><s5><s6><s7><s8>
        token ids: [376, ..., 9162, 19138, 675, 278, 2038, 13382, 2629, 9475, 3838, 29901, 29871, 
                    32008, 32011, 32004, 32013, 32007, 32005, 32002, 32014]

        prompt type2: "{}", predict the following passage within eight words: <s9><s10><s11><s12><s13><s14><s15><s16>
        token ids: [376, ..., 8500, 278, 1494, 13382, 2629, 9475, 3838, 29901, 29871, 32000, 32009, 
                    32012, 32001, 32010, 32003, 32006, 32015]
                    
        [9162, 19138, 675, 278, 2038, 13382, 2629, 9475, 3838, 29901,
        29871, 32008, 32011, 32004, 32013, 32007, 32005, 32002, 32014,
        8500, 278, 1494, 13382, 2629, 9475, 3838, 29901, 29871, 32000,
        32009, 32012, 32001, 32010, 32003, 32006, 32015]
                    
        Maybe only one of them will appear, or both may appear. We consider all possibilities here.
        """

        self.summarize_prompt_ids = [9162, 19138, 675, 278, 2038, 13382, 2629, 9475, 3838, 29901, 29871,
                                     32008, 32011, 32004, 32013, 32007, 32005, 32002, 32014]
        self.predict_prompt_ids = [8500, 278, 1494, 13382, 2629, 9475, 3838, 29901, 29871, 32000, 32009,
                                   32012, 32001, 32010, 32003, 32006, 32015]

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_summarize_ids: Optional[torch.LongTensor] = None,
            output_predict_ids: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            output_ids refers to the set of token IDs to be predicted

        Returns:
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        ar_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            ar_loss = loss_fct(shift_logits, shift_labels)

        """
        prompt: ", summarize the above passage within eight words: <s1><s2><s3><s4><s5><s6><s7><s8>", predict the following passage within eight words: <s9><s10><s11><s12><s13><s14><s15><s16>
        token ids: [9162, 19138, 675, 278, 2038, 13382, 2629, 9475, 3838, 29901,
                    29871, 32008, 32011, 32004, 32013, 32007, 32005, 32002, 32014,
                    8500, 278, 1494, 13382, 2629, 9475, 3838, 29901, 29871, 32000,
                    32009, 32012, 32001, 32010, 32003, 32006, 32015]
        token ids[-25:-17] —— <s1><s2><s3><s4><s5><s6><s7><s8>
        token ids[-8:] —— <s9><s10><s11><s12><s13><s14><s15><s16>
        """
        bow_summarize_loss = 0
        if output_summarize_ids is not None:
            special_logits = logits[:,-len(self.predict_prompt_ids) - 8:-len(self.predict_prompt_ids),:]
            special_logits, _ = torch.max(special_logits, dim=1)
            bow_summarize_loss = 0
            possibility = self.log_softmax(special_logits)
            batch_num = 0
            for p, temp_output_ids in zip(possibility, output_summarize_ids):
                unique_useful_ids = torch.unique(temp_output_ids[temp_output_ids > 2])
                if len(unique_useful_ids) > 0:
                    bow_summarize_loss -= torch.mean(p[unique_useful_ids])
                    batch_num += 1
            if batch_num > 0:
                bow_summarize_loss /= batch_num
                bow_summarize_loss /= 10

        bow_predict_loss = 0
        if output_predict_ids is not None:
            special_logits = logits[:, -8:, :]
            special_logits, _ = torch.max(special_logits, dim=1)
            bow_predict_loss = 0
            possibility = self.log_softmax(special_logits)
            batch_num = 0
            for p, temp_output_ids in zip(possibility, output_predict_ids):
                unique_useful_ids = torch.unique(temp_output_ids[temp_output_ids > 2])
                if len(unique_useful_ids) > 0:
                    bow_predict_loss -= torch.mean(p[unique_useful_ids])
                    batch_num += 1
            if batch_num > 0:
                bow_predict_loss /= batch_num
                bow_predict_loss /= 10

        if bow_summarize_loss > 0 and bow_predict_loss > 0:
            bow_loss = (bow_summarize_loss + bow_predict_loss) / 2
        elif bow_summarize_loss > 0:
            bow_loss = bow_summarize_loss
        elif bow_predict_loss > 0:
            bow_loss = bow_predict_loss
        else:
            bow_loss = None

        if ar_loss is not None and bow_loss is not None:
            loss = ar_loss + bow_loss
        elif ar_loss is None:
            loss = bow_loss
        else:
            loss = ar_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            # if dist.get_rank() == 0:
            #     print(ar_loss, bow_summarize_loss, bow_predict_loss)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PreModel(nn.Module):
    def __init__(self,
                 model: AutoModel = None,
                 ):
        super().__init__()
        self.model = model

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        self.model.enable_input_require_grads(**kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)