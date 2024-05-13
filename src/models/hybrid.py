import torch

from torch import nn, Tensor
from typing import Callable

from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput

from transformers.models.mamba.modeling_mamba import (
    MambaMixer,
    MambaBlock,
    MambaConfig,
)

from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaAttention,
    LlamaRotaryEmbedding,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaConfig,
)

from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2MLP,
    GPT2Block,
    GPT2Config,
)

from .transformer import BackboneModel


SUPPORTED_BLOCKS = [
    "residual",
    "rms norm",
    "layer norm",
    "absolute positional embedding",

    "llama attention",
    "llama attention no rope",
    "gpt2 attention",
    "mamba mixer",

    "llama mlp",
    "gpt2 mlp",

    "llama block",
    "gpt2 block",
    "mamba block",
]

class ResidualMarker(nn.Module):
    """
    This class is a dummy module to mark where 
    residual connections should be made
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.Identity()
    
    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)


class AbsolutePositionalEmbedding(nn.Module):

    def __init__(self, max_num_positions, hidden_dim):
        super().__init__()
        self.positions = torch.arange(0, max_num_positions, dtype=torch.int)
        self.embed = nn.Embedding(max_num_positions, hidden_dim)

    def forward(self, inp: Tensor) -> Tensor:
        *_, seq_len, _ = inp.shape
        pos = self.positions.to(device=inp.device)
        embeddings = self.embed(pos)[:seq_len]
        return inp + embeddings

class RotaryEmbeddingStub(LlamaRotaryEmbedding):
    def __init__(self, *args, enable: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        self.enable = enable

    def forward(self, x, position_ids):
        cos, sin = super().forward(x, position_ids)

        if self.enable:
            return cos, sin
        return torch.ones_like(cos), torch.zeros_like(sin)

def _make_llama_attention_factory(config: PretrainedConfig, layer_idx: int, use_rope: bool) -> Callable[[], nn.Module]:
    def llama_attn_factory():
        attn_module = LlamaAttention(config=config, layer_idx=layer_idx) # pyright: ignore[reportArgumentType]
        old_posemb = attn_module.rotary_emb

        attn_module.rotary_emb = RotaryEmbeddingStub(
            dim=old_posemb.dim,
            max_position_embeddings=old_posemb.max_position_embeddings,
            base=old_posemb.base,
            scaling_factor=old_posemb.scaling_factor,
            device=None, # TODO: replace with proper value
            enable=use_rope
        )

        return attn_module

    return llama_attn_factory

def SPEC_TO_MODULE(spec_name: str, config: PretrainedConfig, layer_idx: int) -> nn.Module:
    # We wrap the instantiation of each of the modules in a lambda
    # construct to avoid instantiating every layer whenever we call
    # this mapping
    MAPPING: dict[str, Callable[[], nn.Module]] = {
        "residual"   : lambda: ResidualMarker(),
        "rms norm"   : lambda: LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps), # MambaRMSNorm is identical
        "layer norm" : lambda: nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon),
        "absolute positional embedding" : lambda: AbsolutePositionalEmbedding(config.max_position_embeddings, config.hidden_size),

        "llama attention"  : _make_llama_attention_factory(config, layer_idx, use_rope=True),
        "llama attention no rope" : _make_llama_attention_factory(config, layer_idx, use_rope=False),
        "gpt2 attention"   : lambda: GPT2Attention(config=config, layer_idx=layer_idx),
        "mamba mixer" : lambda: MambaMixer(config, layer_idx=layer_idx),

        "llama mlp" : lambda: LlamaMLP(config),
        "gpt2 mlp"  : lambda: GPT2MLP(config.n_inner if config.n_inner is not None else 4 * config.hidden_size, config),

        "llama block" : lambda: LlamaDecoderLayer(config, layer_idx), # pyright: ignore[reportArgumentType]
        "gpt2 block"  : lambda: GPT2Block(config, layer_idx=layer_idx),
        "mamba block" : lambda: MambaBlock(config, layer_idx=layer_idx),
    }

    if not all(key in SUPPORTED_BLOCKS for key in MAPPING.keys()):
        raise Exception("Not all \"supported\" blocks can be instantiated! Make "
                        "sure `MAPPING` in this function and `SUPPORTED_BLOCKS` "
                        "are equal in this file")
    return MAPPING[spec_name]()


class HybridBackbone(nn.Module):

    def has(self, module_substr: str):
        return any(map(lambda mod: module_substr in mod, self.module_names))

    def __init__(
        self, 
        module_names: list[str] = ["residual", "rms norm", "llama attn", "residual", "llama mlp", "residual"],
        n_positions: int = 101,
        embed_dim: int = 128, 
        n_layer: int = 12, 
        n_head: int = 4, 
        hidden_act: str = 'silu', 
        rope_theta: float = 1e4,
        **kwargs
    ):
        super().__init__()

        flatten = lambda lst: sum(map(flatten, lst), []) if isinstance(lst, list) else [lst]
        self.module_names = flatten(module_names)
        if not all(isinstance(mod, str) for mod in self.module_names):
            raise TypeError(f"Module names for HybridBackbone are malformed! Got:\n{self.module_names}")
    
        self.raw_config = {
            "n_positions" : n_positions,
            "n_embd" : embed_dim,
            "n_layer" : n_layer,
            "n_head" : n_head,
            "hidden_act" : hidden_act,
            "rope_theta" : rope_theta,
            **kwargs
        }

        self.llama_config = LlamaConfig(
            max_position_embeddings=2 * n_positions,
            hidden_size=embed_dim,
            intermediate_size=4*embed_dim,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            hidden_act=hidden_act,
            rope_theta=rope_theta,
            use_cache=False, # On inspection, this only writes to cache, not reads(?)
            **kwargs # only provide this config all params to serve as default config later on
        ) if self.has("llama") else None

        self.gpt2_config = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=embed_dim,
            n_layer=n_layer,
            n_head=n_head,
            hidden_act=hidden_act,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        ) if self.has("gpt2") else None

        self.mamba_config = MambaConfig(
            hidden_size=embed_dim,
            num_hidden_layers=n_layer,
                state_size=kwargs.get("mamba_state_size", 16),
                expand=kwargs.get("mamba_expand", 2),
                conv_kernel=kwargs.get("mamba_conv_kernel", 4),
                hidden_act=hidden_act,
            use_cache=False, # we set this to false only for consistency
        ) if self.has("mamba") else None

        modules: list[nn.Module] = [ ]
        for layer_idx, mod_name in enumerate(self.module_names):
            config_for_this_layer = None

            if "llama" in mod_name:
                config_for_this_layer = self.llama_config
            elif "gpt2" in mod_name:
                config_for_this_layer = self.gpt2_config
            elif "mamba" in mod_name:
                config_for_this_layer = self.mamba_config
            else:
                config_for_this_layer = self.llama_config

            if config_for_this_layer is None:
                raise NotImplementedError(f"Failed to load a config for layer: {mod_name}!")

            modules.append(
                SPEC_TO_MODULE(
                    mod_name, 
                    config_for_this_layer, 
                    layer_idx=layer_idx
                )
            )

        self.layers = nn.ModuleList(modules)

    def forward(self, inputs_embeds) -> BaseModelOutput:
        hidden_state = inputs_embeds
        residual = 0

        for layer in self.layers:
            forward_kwargs = { }

            if isinstance(layer, (LlamaAttention, )):
                forward_kwargs.update({ 
                    "position_ids" : torch.arange(
                        0, hidden_state.shape[1],
                        device=hidden_state.device
                    ).unsqueeze(0)
                })

            layer = layer.to(hidden_state.device)
            hidden_state = layer(
                hidden_state,
                **forward_kwargs
            )

            if isinstance(hidden_state, (tuple, )): # collect only attention outputs for attn layers
                hidden_state = hidden_state[0]

            if isinstance(layer, ResidualMarker):
                hidden_state = residual + hidden_state
                residual = hidden_state

        return BaseModelOutput(
            last_hidden_state=hidden_state
        )

class HybridModel(BackboneModel):

    def __init__(self, module_names: list[str], x_dim: int, n_positions: int, n_embd: int = 128, y_dim: int = 1, **kwargs):
        backbone = HybridBackbone(
            module_names, 
            embed_dim=n_embd,
            n_positions=n_positions,
            **kwargs
        )
        super().__init__(backbone, x_dim, n_positions, n_embd, y_dim)

