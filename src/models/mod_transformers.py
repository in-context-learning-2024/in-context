import torch
from transformers import GPT2Config, GPT2Model, MambaConfig, MambaPreTrainedModel, MambaModel # type: ignore
from torch import nn
from .transformer import TransformerModel
from typing import Optional, Tuple, Union
import types
from core import ContextModel

from .mod_seq_model import ModSeqModel
from .attention_fns import vit_style_relu_attn, causal_relu_attn
from .inferencing_fns import forward_GPT2Model, block_var_declare_no_change, forward_block_mod_transformer

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import functools
      
class ModTransformerModel(ModSeqModel):
    def __init__(self, x_dim, n_positions, n_embd=128, n_layer=12, n_head=4, want_pos_embeddings=True, no_attention=False, custom_attn_func=None, num_mamba_layers=1, num_mamba_instances=2, **kwargs):
        super(ModTransformerModel, self).__init__()

        mamba_configuration = MambaConfig(
            vocab_size=gpt_configuration.vocab_size,
            hidden_size=n_embd,
            layer_norm_epsilon=gpt_configuration.layer_norm_epsilon,
            num_hidden_layers=num_mamba_layers,
            use_cache=gpt_configuration.use_cache
        )

        self.name = f"mod_gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
      
        self.change_gpt2_block(block_var_declare_no_change, gpt_configuration, forward_block_mod_transformer)
