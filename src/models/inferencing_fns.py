import torch
from torch import LongTensor, FloatTensor, Tensor
from transformers import GPT2Config, GPT2Model, MambaConfig, MambaPreTrainedModel, MambaModel # pyrigh: ignor[]
from torch import nn
from .transformer import TransformerModel
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

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
# In the following function BLOCK_VAR_DECLARE, one can add new variables, remove unnecessary variables, and modify existing ones.
# BLOCK_VAR_DECLARE should only be called once: at the initialization of your custom model.

# Modify the forward function as seen fit. 
#
# This is a way to modify layer architecture.

def block_var_declare_mambaformer(self, this_mamba_model):
    self.norm_f_1 = this_mamba_model[0].norm_f
    self.norm_f_2 = this_mamba_model[1].norm_f
    self.mamba_blocks_beg = list(this_mamba_model[0].layers)
    self.mamba_blocks_end = list(this_mamba_model[1].layers)
    
#Used for mamba_no_attention and mambafirstformer
def block_var_declare_mamba_single(self, this_mamba_model):
    self.norm_f = this_mamba_model.norm_f
    self.mamba_blocks = list(this_mamba_model.layers)

def block_var_declare_no_change(self, example_var):
  pass

def forward_through_mamba_blocks(hidden_states, these_mamba_blocks, this_norm_f):
    for mb in these_mamba_blocks:
        hidden_states = mb(hidden_states)

    hidden_states = this_norm_f(hidden_states)
    return hidden_states
    
def forward_block_mod_transformer(
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
        
        if not no_attention:
            hidden_states = self.ln_1(hidden_states)
            
            ln_cross_attn = self.ln_cross_attn if hasattr(self, "ln_cross_attn") and self.ln_cross_attn else None
            crossattention = self.crossattention if hasattr(self, "crossattention") else None
            hidden_states, outputs = forward_through_attention(attention_block=self.attn,
                                                               crossattention=crossattention,
                                                               ln_cross_attn=ln_cross_attn,
                                                               residual=residual,
                                                               hidden_states=hidden_states,
                                                               layer_past=layer_past,
                                                               attention_mask=attention_mask,
                                                               head_mask=head_mask,
                                                               use_cache=use_cache,
                                                               output_attentions=output_attentions,
                                                               encoder_hidden_states=encoder_hidden_states,
                                                               encoder_attention_mask=encoder_attention_mask)
                
            residual = hidden_states
        
        hidden_states = self.ln_2(hidden_states)
      
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        return output_processing(outputs, hidden_states, no_attention, use_cache)

def forward_through_attention(attention_block=None,
                              crossattention=None,
                              ln_cross_attn=None,
                              residual=None,
                              hidden_states=None,
                              layer_past=None,
                              attention_mask=None,
                              head_mask=None,
                              use_cache=False,
                              output_attentions=False,
                              encoder_hidden_states=None,
                              encoder_attention_mask=None):
        attn_outputs = attention_block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual
            
        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not crossattention:
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = ln_cross_attn(hidden_states)
            cross_attn_outputs = crossattention(
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

        return hidden_states, outputs
                
def forward_block_mambaformer(
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

        #Utilizing Mamba...

        residual = hidden_states
      
        hidden_states = forward_through_mamba_blocks(hidden_states, self.mamba_blocks_beg, self.norm_f_1)
        
        hidden_states = hidden_states + residual
        
        residual = hidden_states
        
        if not no_attention:
            hidden_states = self.ln_1(hidden_states)
            
            ln_cross_attn = self.ln_cross_attn if hasattr(self, "ln_cross_attn") and self.ln_cross_attn else None
            crossattention = self.crossattention if hasattr(self, "crossattention") else None
            hidden_states, outputs = forward_through_attention(attention_block=self.attn,
                                                               crossattention=crossattention,
                                                               ln_cross_attn=ln_cross_attn,
                                                               residual=residual,
                                                               hidden_states=hidden_states,
                                                               layer_past=layer_past,
                                                               attention_mask=attention_mask,
                                                               head_mask=head_mask,
                                                               use_cache=use_cache,
                                                               output_attentions=output_attentions,
                                                               encoder_hidden_states=encoder_hidden_states,
                                                               encoder_attention_mask=encoder_attention_mask)
                
            residual = hidden_states
        
        hidden_states = self.ln_2(hidden_states)

        hidden_states = forward_through_mamba_blocks(hidden_states, self.mamba_blocks_end, self.norm_f_2)
      
        hidden_states = residual + hidden_states

        return output_processing(outputs, hidden_states, no_attention, use_cache)
        
def forward_block_mambafirstformer(
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

        #Utilizing Mamba...
        hidden_states = forward_through_mamba_blocks(hidden_states, self.mamba_blocks, self.norm_f)
        #The above was all an addition to the vanilla transformer
        
        residual = hidden_states
        
        if not no_attention:
            hidden_states = self.ln_1(hidden_states)
            
            ln_cross_attn = self.ln_cross_attn if hasattr(self, "ln_cross_attn") and self.ln_cross_attn else None
            crossattention = self.crossattention if hasattr(self, "crossattention") else None
            
            hidden_states, outputs = forward_through_attention(attention_block=self.attn,
                                                               crossattention=crossattention,
                                                               ln_cross_attn=ln_cross_attn,
                                                               residual=residual,
                                                               hidden_states=hidden_states,
                                                               layer_past=layer_past,
                                                               attention_mask=attention_mask,
                                                               head_mask=head_mask,
                                                               use_cache=use_cache,
                                                               output_attentions=output_attentions,
                                                               encoder_hidden_states=encoder_hidden_states,
                                                               encoder_attention_mask=encoder_attention_mask)
                
            residual = hidden_states
        
        hidden_states = self.ln_2(hidden_states)

        feed_forward_hidden_states = self.mlp(hidden_states)

        hidden_states = residual + feed_forward_hidden_states

        return output_processing(outputs, hidden_states, no_attention, use_cache)
        
def forward_block_mamba_no_attention(
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

        hidden_states = forward_through_mamba_blocks(hidden_states, self.mamba_blocks, self.norm_f)

        hidden_states = residual + hidden_states

        residual = hidden_states
      
        if not no_attention:
            hidden_states = self.ln_1(hidden_states)
            
            ln_cross_attn = self.ln_cross_attn if hasattr(self, "ln_cross_attn") and self.ln_cross_attn else None
            crossattention = self.crossattention if hasattr(self, "crossattention") else None
            hidden_states, outputs = forward_through_attention(attention_block=self.attn,
                                                               crossattention=crossattention,
                                                               ln_cross_attn=ln_cross_attn,
                                                               residual=residual,
                                                               hidden_states=hidden_states,
                                                               layer_past=layer_past,
                                                               attention_mask=attention_mask,
                                                               head_mask=head_mask,
                                                               use_cache=use_cache,
                                                               output_attentions=output_attentions,
                                                               encoder_hidden_states=encoder_hidden_states,
                                                               encoder_attention_mask=encoder_attention_mask)
                
            residual = hidden_states
        

        feed_forward_hidden_states = self.mlp(hidden_states)

        hidden_states = residual + feed_forward_hidden_states

        return output_processing(outputs, hidden_states, no_attention, use_cache)  # hidden_states, present, (attentions, cross_attentions)

def output_processing(outputs, hidden_states, no_attention, use_cache):
    if no_attention: 
        outputs = (hidden_states,)
    elif use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]
    return outputs
    
def forward_GPT2Model(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        no_attention = False,
        want_pos_embeddings = True
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else (None if no_attention else self.config.output_attentions)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if not no_attention:
            if past_key_values is None:
                past_length = 0
                past_key_values = tuple([None] * len(self.h))
            else:
                past_length = past_key_values[0][0].size(-2)
        else:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
                
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # GPT2Attention mask.
        if not no_attention and attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if not no_attention and self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if not no_attention:
            head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        if want_pos_embeddings:
            hidden_states = inputs_embeds + position_embeds
        else:
            hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions and not no_attention else None
        all_cross_attentions = () if output_attentions and not no_attention and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if not no_attention and attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if not no_attention and isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            #print(self.gradient_checkpointing)
            #print(self.training)
            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    None if no_attention else attention_mask,
                    None if no_attention else head_mask[i],
                    encoder_hidden_states,
                    None if no_attention else encoder_attention_mask,
                    use_cache,
                    None if no_attention else output_attentions
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask= None if no_attention else attention_mask,
                    head_mask= None if no_attention else head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None if no_attention else encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=None if no_attention else output_attentions
                )
                #print(outputs)

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions and not no_attention:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )