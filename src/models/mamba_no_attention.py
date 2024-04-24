import torch
from torch import LongTensor, FloatTensor, Tensor
from transformers import GPT2Config, GPT2Model, MambaConfig, MambaPreTrainedModel, MambaModel # pyrigh: ignor[]
from torch import nn
from .transformer import TransformerModel
from typing import Optional, Tuple, Union
import types
from core import ContextModel

from .attention_fns import vit_style_relu_attn, causal_relu_attn
from .inferencing_fns import forward_GPT2Model

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import functools



# FOR REFERENCE: A GPT2 Block is initialized with the following code:

# def __init__(self, config, layer_idx=None, attn_func=None):
#         super().__init__()
#         hidden_size = config.hidden_size
#         inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

#         self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
#         self.attn = GPT2Attention(config, layer_idx=layer_idx, attn_func=attn_func)
#         self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

#         if config.add_cross_attention:
#             self.crossattention = GPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx, attn_func=attn_func)
#             self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

#         self.mlp = GPT2MLP(inner_dim, config)
# ----------------------------------------------------------------------
# In the following function BLOCK VAR DECLARE, one can add new variables, remove unnecessary variables, and modify existing ones.
# Then, modify the forward function as seen fit. 
#
# This is a way to modify layer architecture.

#no cache
def block_var_declare(self, this_mamba_model):
    self.norm_f = this_mamba_model.norm_f
    self.mamba_blocks = list(this_mamba_model.layers)
    
def forward_block(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        no_attention = False
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        
        residual = hidden_states
      
        for mb in self.mamba_blocks:
            hidden_states = mb(hidden_states)

        hidden_states = self.norm_f(hidden_states)

        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.ln_1(hidden_states)
      
        if not no_attention:
            #for _ in range(1000):
            #   print ("LKSDJFOSJFOIWJFIOWEJFOIWEJFOIEWJFOIWEJFOIWEFJOIWJ")
            hidden_states = self.ln_1(hidden_states)
    
            attn_outputs = self.attn(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            #print(attn_outputs)
            attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
            outputs = attn_outputs[1:]
            # residual connection
            hidden_states = attn_output + residual
            
            if encoder_hidden_states is not None and not no_attention:
                # add one self-attention block for cross-attention
                if not hasattr(self, "crossattention"):
                    raise ValueError(
                        f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                        "cross-attention layers by setting `config.add_cross_attention=True`"
                    )
                residual = hidden_states
                hidden_states = self.ln_cross_attn(hidden_states)
                cross_attn_outputs = self.crossattention(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                attn_output = cross_attn_outputs[0]
                # residual connection
                hidden_states = residual + attn_output
                outputs = outputs + cross_attn_outputs[2:] 
            #print(outputs)
            #print("NEWJROWFJWIOFJIO") # add cross attentions if we output attention weights
                
            residual = hidden_states
        

        feed_forward_hidden_states = self.mlp(hidden_states)

        hidden_states = residual + feed_forward_hidden_states

        if no_attention: 
            #Might cause errors if output of attention does not make OUTPUTS a list
            outputs = (hidden_states,)
        elif use_cache:
            #print(outputs)
            outputs = (hidden_states,) + outputs
            #print("AFTERHIDDEN")
            #print(outputs)
        else:
            #print(outputs)
            #print(outputs[1:])
            outputs = (hidden_states,) + outputs[1:]
            #print(outputs)
            #print("AFTERHIDDEN")
            #print(outputs.shape)
        
        
        
        return outputs  # hidden_states, present, (attentions, cross_attentions)
        
      
class MambaNoAttentionModel(ContextModel):
    def __init__(self, x_dim, n_positions, n_embd=128, n_layer=12, n_head=4, want_pos_embeddings=True, no_attention=False, custom_attn_func=None, num_mamba_layers=1, **kwargs):
        super(MambaNoAttentionModel, self).__init__()
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
            num_hidden_layers=1,
            use_cache=gpt_configuration.use_cache
        )
        
        self.name = f"mamba_embd={n_embd}_layer={n_layer}"
        
        if custom_attn_func == "relu":
            self.custom_attn_func = relu_attn
        elif custom_attn_func == "relu_causal":
            self.custom_attn_func = relu_attn_causal
        else:
            self.custom_attn_func = None

        self.context_length = n_positions
        self._n_dims = x_dim
        self._read_in = nn.Linear(x_dim, n_embd)
      
        self._backbone = GPT2Model(gpt_configuration)

        #Allow for attention and pos embeddings
        self._backbone.forward = types.MethodType(functools.partial(forward_GPT2Model, no_attention=no_attention, want_pos_embeddings=want_pos_embeddings), self._backbone)
        
        for x in list(self._backbone.children())[3]:
            x.forward = types.MethodType(functools.partial(forward_block, no_attention=no_attention), x)
            block_var_declare(x, MambaModel(mamba_configuration))
        
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
