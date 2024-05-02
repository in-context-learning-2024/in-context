import torch
from torch import LongTensor, FloatTensor, Tensor
from transformers import GPT2Config, GPT2Model, MambaConfig, MambaPreTrainedModel, MambaModel # pyrigh: ignor[]
from torch import nn
from .transformer import TransformerModel
from typing import Optional, Tuple, Union
import types
from core import ContextModel

from .mod_seq_model import ModSeqModel
from .inferencing_fns import block_var_declare_mamba_single, forward_block_mamba_no_attention, forward_block_mambafirstformer, block_var_declare_mambaformer, forward_block_mambaformer, block_var_declare_no_change, forward_block_mod_transformer


class MambaNoAttentionModel(ModSeqModel):
    def __init__(self, x_dim, n_positions, n_embd=128, n_layer=12, n_head=4, want_pos_embeddings=True, no_attention=False, custom_attn_func=None, num_mamba_layers=1, **kwargs):
        super(MambaNoAttentionModel, self).__init__(x_dim, n_positions, n_embd=n_embd, n_layer=n_layer, n_head=n_head, want_pos_embeddings=want_pos_embeddings, no_attention=no_attention, custom_attn_func=custom_attn_func)
        mamba_configuration = MambaConfig(
            vocab_size=self.gpt2_configuration.vocab_size,
            hidden_size=n_embd,
            layer_norm_epsilon=self.gpt2_configuration.layer_norm_epsilon,
            num_hidden_layers=num_mamba_layers,
            use_cache=self.gpt2_configuration.use_cache
        )
        
        self.name = f"mamba_seq_embd={n_embd}_layer={n_layer}"

        self.change_gpt2_block(block_var_declare_mamba_single, MambaModel(mamba_configuration), forward_block_mamba_no_attention)

        
class MambaFirstGPT2TransformerModel(ModSeqModel):
    def __init__(self, x_dim, n_positions, n_embd=128, n_layer=12, n_head=4, want_pos_embeddings=True, no_attention=False, custom_attn_func=None, num_mamba_layers=1, **kwargs):
        super(MambaFirstGPT2TransformerModel, self).__init__(x_dim, n_positions, n_embd=n_embd, n_layer=n_layer, n_head=n_head, want_pos_embeddings=want_pos_embeddings, no_attention=no_attention, custom_attn_func=custom_attn_func)
          
        mamba_configuration = MambaConfig(
            vocab_size=self.gpt2_configuration.vocab_size,
            hidden_size=n_embd,
            layer_norm_epsilon=self.gpt2_configuration.layer_norm_epsilon,
            num_hidden_layers=num_mamba_layers,
            use_cache=self.gpt2_configuration.use_cache
        )
        

        self.name = f"mambafirstgpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.change_gpt2_block(block_var_declare_mamba_single, MambaModel(mamba_configuration), forward_block_mambafirstformer)

class MambaformerModel(ModSeqModel):
    def __init__(self, x_dim, n_positions, n_embd=128, n_layer=12, n_head=4, want_pos_embeddings=True, no_attention=False, custom_attn_func=None, num_mamba_layers=1, num_mamba_instances=2, **kwargs):
        super(MambaformerModel, self).__init__(x_dim, n_positions, n_embd=n_embd, n_layer=n_layer, n_head=n_head, want_pos_embeddings=want_pos_embeddings, no_attention=no_attention, custom_attn_func=custom_attn_func)
       
        mamba_configuration = MambaConfig(
            vocab_size=self.gpt2_configuration.vocab_size,
            hidden_size=n_embd,
            layer_norm_epsilon=self.gpt2_configuration.layer_norm_epsilon,
            num_hidden_layers=num_mamba_layers,
            use_cache=self.gpt2_configuration.use_cache
        )
        

        self.name = f"mambaformer_embd={n_embd}_layer={n_layer}_head={n_head}"
        
        self.change_gpt2_block(block_var_declare_mambaformer, [MambaModel(mamba_configuration) for _ in range(num_mamba_instances)], forward_block_mambaformer)
      
class ModTransformerModel(ModSeqModel):
    def __init__(self, x_dim, n_positions, n_embd=128, n_layer=12, n_head=4, want_pos_embeddings=True, no_attention=False, custom_attn_func=None, num_mamba_layers=1, num_mamba_instances=2, **kwargs):
        super(ModTransformerModel, self).__init__(x_dim, n_positions, n_embd=n_embd, n_layer=n_layer, n_head=n_head, want_pos_embeddings=want_pos_embeddings, no_attention=no_attention, custom_attn_func=custom_attn_func)

        self.name = f"mod_gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
      
        self.change_gpt2_block(block_var_declare_no_change, self.gpt2_configuration, forward_block_mod_transformer)
