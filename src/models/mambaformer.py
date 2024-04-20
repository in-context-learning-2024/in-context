import torch
from transformers import GPT2Config, GPT2Model, MambaConfig, MambaPreTrainedModel, MambaModel # type: ignore
from torch import nn
from .transformer import TransformerModel
from typing import Optional, Tuple, Union
import types
from core import ContextModel

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import functools

def relu_attn(self, query, key, value, attention_mask=None, head_mask=None):
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

def relu_attn_causal(self, query, key, value, attention_mask=None, head_mask=None):
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
  
      # TODO: make this sequence length causal (divide by tokens seen so far, not total tokens in sequence)
      # relud = nn.functional.relu(attn_weights)
    seq_len = query.size(-2)
    causal_seq_len = 1 + ( torch.arange(seq_len, device=DEVICE)
                                  .expand(attn_weights.shape)
                                  .transpose(-1, -2) )
      # import code
      # assert attn_weights.shape == causal_seq_len.shape, code.interact(local=locals(), banner=f"Failed shape check: attn_weights do not math causal_seq_len in shape! \n{attn_weights.shape} vs {causal_seq_len.shape}")
      # pre_attn_weights = attn_weights
    attn_weights = nn.functional.relu(attn_weights) / (causal_seq_len + 1)
      # code.interact(local=locals(), banner="yeesh")
  
      # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)
  
      # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask
  
    attn_output = torch.matmul(attn_weights, value)
  
    return attn_output, attn_weights

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
    self.norm_f_1 = this_mamba_model[0].norm_f
    self.norm_f_2 = this_mamba_model[1].norm_f
    self.mamba_blocks_beg = list(this_mamba_model[0].layers)
    self.mamba_blocks_end = list(this_mamba_model[1].layers)

    
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

        #Utilizing Mamba...
        residual = hidden_states
      
        for mb in self.mamba_blocks_beg:
            hidden_states = mb(hidden_states)

        hidden_states = self.norm_f_1(hidden_states)

        hidden_states = hidden_states + residual
        
        residual = hidden_states
        
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
        
        hidden_states = self.ln_2(hidden_states)

        for mb in self.mamba_blocks_end:
            hidden_states = mb(hidden_states)

        hidden_states = self.norm_f_2(hidden_states)
      
        hidden_states = residual + hidden_states

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
        
      
class MambaformerModel(ContextModel):
    def __init__(self, x_dim, n_positions, n_embd=128, n_layer=12, n_head=4, want_pos_embeddings=True, no_attention=False, custom_attn_func=None, num_mamba_layers=1, num_mamba_instances=2, **kwargs):
        super(MambaformerModel, self).__init__()
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
            self.custom_attn_func = relu_attn
        elif custom_attn_func == "relu_causal":
            self.custom_attn_func = relu_attn_causal
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
            x.forward = types.MethodType(functools.partial(forward_block, no_attention=no_attention), x)
            block_var_declare(x, [MambaModel(mamba_configuration) for _ in range(num_mamba_instances)])

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
