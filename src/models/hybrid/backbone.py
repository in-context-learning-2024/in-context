from typing import Any, Callable

import torch

from torch import nn, Tensor
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.mamba.modeling_mamba import MambaConfig
from transformers.models.llama.modeling_llama import LlamaConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Config

from .layers import SPEC_TO_MODULE, LlamaAttention, ResidualMarker

class HybridBackbone(nn.Module):

    def has(self, module_substr: str):
        return any(map(lambda mod: module_substr in mod, self.module_names))

    def __init__(
        self, 
        module_names: list[str],
        n_positions: int,
        embed_dim: int, 
        n_head: int,
        rope_theta: float = 1e4,
        **kwargs: Any
    ):
        super().__init__()

        flatten = lambda lst: sum(map(flatten, lst), []) if isinstance(lst, list) else [lst]
        self.module_names = flatten(module_names)
        if not all(isinstance(mod, str) for mod in self.module_names):
            raise TypeError(f"Module names for HybridBackbone are malformed! Got:\n{self.module_names}")

        n_layer = len(self.module_names)
        self.raw_config = {
            "n_positions" : n_positions,
            "n_embd" : embed_dim,
            "n_layer" : n_layer,
            "n_head" : n_head,
            "rope_theta" : rope_theta,
            **kwargs
        }

        self.layer_configs: dict[str, Callable[[], PretrainedConfig]] = {
            "llama" : lambda: LlamaConfig(
                max_position_embeddings=2 * n_positions,
                hidden_size=embed_dim,
                intermediate_size=4*embed_dim,
                num_hidden_layers=n_layer,
                num_attention_heads=n_head,
                hidden_act=kwargs.get("llama_hidden_act", "silu"),
                rope_theta=rope_theta,
                use_cache=False, # On inspection, this only writes to cache, not reads(?)
                **kwargs # provide all params to only this config to serve as default later on
            ),
            "mamba" : lambda: MambaConfig(
                hidden_size=embed_dim,
                num_hidden_layers=n_layer,
                state_size=kwargs.get("mamba_state_size", 16),
                expand=kwargs.get("mamba_expand", 4),
                conv_kernel=kwargs.get("mamba_conv_kernel", 4),
                hidden_act=kwargs.get("mamba_hidden_act", "silu"),
                use_cache=False, # we set this to false only for consistency
            ),
            "gpt2" : lambda: GPT2Config(
                n_positions=2 * n_positions,
                n_embd=embed_dim,
                n_layer=n_layer,
                n_head=n_head,
                activation_function=kwargs.get("gpt2_hidden_act", "gelu_new"),
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
                use_cache=False,
            )
        }

        # instantiate the modules with their respective configurations
        modules: list[nn.Module] = [ ]
        for layer_idx, mod_name in enumerate(self.module_names):
            config_for_this_layer = None

            config_for_this_layer = self.layer_configs["llama"]()
            for search_string, config_constructor in self.layer_configs.items():
                if search_string in mod_name:
                    config_for_this_layer = config_constructor()
                    break

            if config_for_this_layer is None:
                raise NotImplementedError(f"Failed to load a config for layer: {mod_name}!")

            try:
                mod = SPEC_TO_MODULE(
                    mod_name, 
                    config_for_this_layer, 
                    layer_idx=layer_idx
                )
            except TypeError as e:
                raise TypeError(f"Invalid arguments!: {e}") from e

            modules.append(mod)

        self.layers = nn.ModuleList(modules)

    def forward(self, inputs_embeds: Tensor) -> BaseModelOutput:
        hidden_state = inputs_embeds
        residual = 0
        attention_mask = torch.triu( # TODO: remove if possible/reasonable
            torch.full(
                ( # bsz, n_heads, seq_len, seq_len
                    inputs_embeds.shape[0], self.raw_config['n_head'],
                    inputs_embeds.shape[1], inputs_embeds.shape[1]
                ),
                fill_value=torch.finfo(inputs_embeds.dtype).min,
                device=inputs_embeds.device
            ),
            diagonal=1
        )

        for layer in self.layers:
            forward_kwargs: dict[str, Any] = { }

            if isinstance(layer, (LlamaAttention, )): # TODO: remove somehow
                forward_kwargs.update({ 
                    "position_ids" : torch.arange(
                        0, hidden_state.shape[1],
                        device=hidden_state.device
                    ).unsqueeze(0),
                    "attention_mask" : attention_mask,
                })

            layer = layer.to(hidden_state.device)
            hidden_state: tuple[Tensor, ...] | Tensor = layer(
                hidden_state,
                **forward_kwargs
            )

            if isinstance(hidden_state, (tuple, )): # collect only attention outputs for attn layers
                hidden_state = hidden_state[0]

            if isinstance(layer, ResidualMarker):
                hidden_state = residual + hidden_state
                residual = hidden_state

        return BaseModelOutput(
            last_hidden_state=hidden_state # pyright: ignore[reportArgumentType]
        )
