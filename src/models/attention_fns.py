import torch
from torch import LongTensor, FloatTensor, Tensor
from transformers import GPT2Config, GPT2Model, MambaConfig, MambaPreTrainedModel, MambaModel # pyrigh: ignor[]
from torch import nn
from .transformer import TransformerModel
from typing import Optional, Tuple, Union
import types
from core import ContextModel

def causal_relu_attn(self, query, key, value, attention_mask=None, head_mask=None):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))
  
    if self.scale_attn_weights:
        attn_weights = attn_weights / (value.size(-1) ** 0.5)
  
    # Layer-wise attention scaling
    if self.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(self.layer_idx + 1)
  
    if not self.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
        attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))
  
    if attention_mask is not None:
        # Apply the attention mask
        attn_weights = attn_weights + attention_mask
  
    seq_len = query.size(-2)
    causal_seq_len = 1 + ( torch.arange(seq_len, device=DEVICE)
                                  .expand(attn_weights.shape)
                                  .transpose(-1, -2) )
    attn_weights = nn.functional.relu(attn_weights) / (causal_seq_len + 1)
  
    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)
  
    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask
  
    attn_output = torch.matmul(attn_weights, value)
  
    return attn_output, attn_weights


def vit_style_relu_attn(self, query, key, value, attention_mask=None, head_mask=None):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))
  
    if self.scale_attn_weights:
        attn_weights = attn_weights / (value.size(-1) ** 0.5)
  
    # Layer-wise attention scaling
    if self.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(self.layer_idx + 1)
  
    if not self.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
        attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))
  
    if attention_mask is not None:
        # Apply the attention mask
        attn_weights = attn_weights + attention_mask
  
    attn_weights = nn.functional.relu(attn_weights) / query.size(-2) 
  
    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)
  
    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask
  
    attn_output = torch.matmul(attn_weights, value)
  
    return attn_output, attn_weights
