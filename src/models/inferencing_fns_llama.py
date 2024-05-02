import torch
from torch import LongTensor, FloatTensor, Tensor
from transformers import LlamaModel, LlamaConfig, MambaConfig, MambaPreTrainedModel, MambaModel # pyrigh: ignor[]
from torch import nn
from .transformer import TransformerModel
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

# FOR REFERENCE: A LlamaDecoderLayer is initialized with the following code:

#  def __init__(self, config: LlamaConfig, layer_idx: int):
#         super().__init__()
#         self.hidden_size = config.hidden_size

#         self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

#         self.mlp = LlamaMLP(config)
#         self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

# ----------------------------------------------------------------------
# In the following function BLOCK_VAR_DECLARE, one can add new variables, remove unnecessary variables, and modify existing ones.
# BLOCK_VAR_DECLARE should only be called once: at the initialization of your custom model.

# Modify the forward function as seen fit. 
#
# This is a way to modify layer architecture.

def block_var_declare_llamamamba(self, this_mamba_model):
    self.norm_f_1 = this_mamba_model[0].norm_f
    self.norm_f_2 = this_mamba_model[1].norm_f
    self.mamba_blocks_beg = list(this_mamba_model[0].layers)
    self.mamba_blocks_end = list(this_mamba_model[1].layers)
    

def forward_through_mamba_blocks(hidden_states, these_mamba_blocks, this_norm_f):
    for mb in these_mamba_blocks:
        hidden_states = mb(hidden_states)

    hidden_states = this_norm_f(hidden_states)
    return hidden_states


def forward_block_llamamamba(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = forward_through_mamba_blocks(hidden_states, self.mamba_blocks_beg, self.norm_f_1)

        hidden_states = hidden_states + residual

        residual = hidden_states
      
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states = forward_through_mamba_blocks(hidden_states, self.mamba_blocks_end, self.norm_f_2)

        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
