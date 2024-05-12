import torch
from torch import LongTensor, FloatTensor, Tensor
from transformers import GPT2Config, GPT2Model, MambaConfig, MambaPreTrainedModel, MambaModel # pyrigh: ignor[]
from torch import nn
from .transformer import BackboneModel, GPT2, Llama
from typing import Optional, Tuple, Union
import types
from core import ContextModel

from .attention_fns import vit_style_relu_attn, causal_relu_attn, forward_llama_attention_standard
from .inferencing_fns import forward_GPT2Model, block_var_declare_mamba_single, forward_block_mamba_no_attention

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import functools


class ModSeqModel(GPT2):
    def __init__(self, x_dim, n_positions, n_embd=128, n_layer=12, n_head=4, want_pos_embeddings=True, no_attention=False, custom_attn_func=None, **kwargs):
        super().__init__(x_dim, n_positions, n_embd=n_embd, n_layer=n_layer, n_head=n_head)
        if custom_attn_func == "relu":
            self.custom_attn_func = vit_style_relu_attn
        elif custom_attn_func == "relu_causal":
            self.custom_attn_func = causal_relu_attn
        else:
            self.custom_attn_func = None

        self.no_attention = no_attention
        self.want_pos_embeddings = want_pos_embeddings
        self._n_dims = x_dim
        
        #Allow for attention and pos embeddings in GPT2Model Forward function
        self._backbone.forward = types.MethodType(functools.partial(forward_GPT2Model, no_attention=no_attention, want_pos_embeddings=want_pos_embeddings), self._backbone)
       
        # Allow for changes in Attention function for GPT2Attention
        if not no_attention:
            self.change_gpt2_attention()

    def change_gpt2_block(self, instantiate_var_fn, instantiate_var_arg, instantiate_forward_fn):
       for x in list(self._backbone.children())[3]:
            x.forward = types.MethodType(functools.partial(instantiate_forward_fn, no_attention=self.no_attention), x)
            instantiate_var_fn(x, instantiate_var_arg)

    def change_gpt2_attention(self):
      if self.custom_attn_func:
            attn_layers = list(self._backbone.children())[3]
            attn_module_class = list(attn_layers[0].children())[1].__class__

            for i in range(len(attn_layers)):
                list(attn_layers[i].children())[1]._attn = self.custom_attn_func.__get__(
                    list(attn_layers[i].children())[1],
                    attn_module_class
                )

class ModSeqModelLlama(Llama):
    def __init__(self, x_dim, n_positions, n_embd=128, n_layer=12, n_head=4, want_rope=True, custom_attn_func=None, hidden_act="silu", rope_theta=1e4, **kwargs):
        super().__init__(x_dim, n_positions, n_embd=n_embd, n_layer=n_layer, n_head=n_head, hidden_act=hidden_act, rope_theta=rope_theta)
        
        if custom_attn_func == None:
            self.attention_fn_template = forward_llama_attention_standard
        
        self.custom_attn_func = functools.partial(self.attention_fn_template, want_rope=want_rope)

        self.want_rope = want_rope
        self._n_dims = x_dim

        for x in list((list(self._backbone.children())[1])):
            x.self_attn.forward = types.MethodType(self.custom_attn_func, x.self_attn)
        

    def change_llama_block(self, instantiate_var_fn, instantiate_var_arg, instantiate_forward_fn):
        # Obtain the individual decoder layers, then replace attention, call instantiate_var, replace block_forward
        for x in list((list(self._backbone.children())[1])):
            #EX: forward_block_llamamamba -> instantiate_forward_fn
            x.forward = types.MethodType(functools.partial(instantiate_forward_fn), x)
            #EX: Block_var_declare_llamamamba
            instantiate_var_fn(x, instantiate_var_arg)
