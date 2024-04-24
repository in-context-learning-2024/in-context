import torch
from torch import LongTensor, FloatTensor, Tensor
from transformers import GPT2Config, GPT2Model, MambaConfig, MambaPreTrainedModel, MambaModel # pyrigh: ignor[]
from torch import nn
from .transformer import TransformerModel
from typing import Optional, Tuple, Union
import types
from core import ContextModel

from .attention_fns import vit_style_relu_attn, causal_relu_attn
from .inferencing_fns import forward_GPT2Model, block_var_declare_mamba_single, forward_block_mamba_no_attention

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import functools


class ModSeqModel(ContextModel):
    def __init__(self, x_dim, n_positions, n_embd=128, n_layer=12, n_head=4, want_pos_embeddings=True, no_attention=False, custom_attn_func=None, **kwargs):
        super(ModSeqModel, self).__init__()
        gpt_configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False
        )
        
        if custom_attn_func == "relu":
            self.custom_attn_func = vit_style_relu_attn
        elif custom_attn_func == "relu_causal":
            self.custom_attn_func = causal_relu_attn
        else:
            self.custom_attn_func = None

        self.no_attention = no_attention
        self.want_pos_embeddings = want_pos_embeddings
        
        self.context_length = n_positions
        self._n_dims = x_dim
        self._read_in = nn.Linear(x_dim, n_embd)
        #Allow for attention and pos embeddings in GPT2Model Forward function
        self._backbone.forward = types.MethodType(functools.partial(forward_GPT2Model, no_attention=no_attention, want_pos_embeddings=want_pos_embeddings), self._backbone)
       
        # Allow for changes in Attention function for GPT2Attention
        self.change_gpt2_attention()
      
        self._read_out = nn.Linear(n_embd, 1)

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

    def forward(self, xs, ys):
        inds = torch.arange(ys.shape[1])
        
        zs = ContextModel.interleave(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs
