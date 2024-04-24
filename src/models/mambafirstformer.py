import torch
from transformers import GPT2Config, GPT2Model, MambaConfig, MambaPreTrainedModel, MambaModel # type: ignore
from torch import nn
from .transformer import TransformerModel
from typing import Optional, Tuple, Union
import types
from core import ContextModel

from .attention_fns import vit_style_relu_attn, causal_relu_attn
from .inferencing_fns import forward_GPT2Model, block_var_declare_mamba_single, forward_block_mambafirstformer

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import functools

      
class MambaFirstGPT2TransformerModel(ContextModel):
    def __init__(self, x_dim, n_positions, n_embd=128, n_layer=12, n_head=4, want_pos_embeddings=True, no_attention=False, custom_attn_func=None, num_mamba_layers=1, **kwargs):
        super(MambaFirstGPT2TransformerModel, self).__init__()
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

        #can't set return dict parameter in mambaconfig for some reason...

        mamba_configuration = MambaConfig(
            vocab_size=gpt_configuration.vocab_size,
            hidden_size=n_embd,
            layer_norm_epsilon=gpt_configuration.layer_norm_epsilon,
            num_hidden_layers=num_mamba_layers,
            use_cache=gpt_configuration.use_cache
        )

        #print("WantPosEmbeddings" + str(want_pos_embeddings))
        #print("No Attention" + str(no_attention))

        self.name = f"mod_gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        
        if custom_attn_func == "relu":
            self.custom_attn_func = vit_style_relu_attn
        elif custom_attn_func == "relu_causal":
            self.custom_attn_func = causal_relu_attn
        else:
            self.custom_attn_func = None

        self.context_length = n_positions
        self._n_dims = x_dim
        self._read_in = nn.Linear(x_dim, n_embd)
      
        #self._backbone = GPT2Model(configuration, attn_func=relu_attn)

        #Patch that i don't really want to go with in the end
        self._backbone = GPT2Model(gpt_configuration)

        #Allow for attention and pos embeddings
        self._backbone.forward = types.MethodType(functools.partial(forward_GPT2Model, no_attention=no_attention, want_pos_embeddings=want_pos_embeddings), self._backbone)
        
        for x in list(self._backbone.children())[3]:
            x.forward = types.MethodType(functools.partial(forward_block_mambafirstformer, no_attention=no_attention), x)
            block_var_declare_mamba_single(x, MambaModel(mamba_configuration))

        #######DEBUGGING
        # print([type(x) for x in self._backbone.children()])
        # print("Additionally")
        # print(list(self._backbone.children())[3])
        ###########
        
        if self.custom_attn_func:
            attn_layers = list(self._backbone.children())[3]
            attn_module_class = list(attn_layers[0].children())[1].__class__

            for i in range(len(attn_layers)):
                list(attn_layers[i].children())[1]._attn = self.custom_attn_func.__get__(
                    list(attn_layers[i].children())[1],
                    attn_module_class
                )

        self._read_out = nn.Linear(n_embd, 1)

    def forward(self, xs, ys):
        inds = torch.arange(ys.shape[1])
        
        zs = ContextModel.interleave(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs
