from torch import nn, Tensor

from transformers import PretrainedConfig

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
    # "rope",

    "llama attn",
    "gpt2 attn",
    "mamba mixer",

    "llama mlp",
    "gpt2 mlp",

    "llama block",
    "gpt2 block",
    "mamba block",
]

def SPEC_TO_MODULE(spec_name: str, config: PretrainedConfig, layer_idx: int) -> nn.Module:
    # We wrap the instantiation of each of the modules in a lambda
    # construct to avoid instantiating every layer whenever we call
    # this mapping
    MAPPING = {
        "residual"   : lambda: ResidualMarker(),
        "rms norm"   : lambda: LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps), # MambaRMSNorm is identical
        "layer norm" : lambda: nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon),
        # "rope"       : LlamaRotaryEmbedding(),

        "llama attn"  : lambda: LlamaAttention(config=config, layer_idx=layer_idx), # pyright: ignore[reportArgumentType]
        "gpt2 attn"   : lambda: GPT2Attention(config=config, layer_idx=layer_idx),
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


class HybridBackbone(nn.Module):

    def has(self, module_substr: str):
        return any(map(lambda mod: module_substr in mod, self.module_spec))

    def __init__(
        self, 
        module_names: list[str],
        n_positions: int,
        n_embd=128, 
        n_layer=12, 
        n_head=4, 
        hidden_act: str = 'silu', 
        rope_theta: float = 1e4,
        **kwargs
    ):
        super().__init__()

        self.module_names = module_names
        self.raw_config = {
            "n_positions" : n_positions,
            "n_embd" : n_embd,
            "n_layer" : n_layer,
            "n_head" : n_head,
            "hidden_act" : hidden_act,
            "rope_theta" : rope_theta,
            **kwargs
        }

        self.llama_config = LlamaConfig(
            max_position_embeddings=2 * n_positions,
            hidden_size=n_embd,
            intermediate_size=4*n_embd,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            hidden_act=hidden_act,
            rope_theta=rope_theta,
            use_cache=False, # On inspection, this only writes to cache, not reads(?)
        ) if self.has("llama") else None

        self.gpt2_config = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            hidden_act=hidden_act,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        ) if self.has("gpt2") else None

        self.mamba_config = MambaConfig(
            # vocab_size=..., # This is never used
            hidden_size=n_embd,
            num_hidden_layers=n_layer,
                state_size=kwargs.get("mamba_state_size", 16),
                expand=kwargs.get("mamba_expand", 2),
                conv_kernel=kwargs.get("mamba_conv_kernel", 4),
                hidden_act=hidden_act,
            use_cache=self.gpt2_configuration.use_cache,
            use_cache=self.llama_configuration.use_cache
        ) if self.has("mamba") else None

        modules: list[nn.Module] = [ ]
        for layer_idx, mod_name in enumerate(module_names):
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

    def forward(self, input_embeds) -> Tensor:
        hidden_state = input_embeds
        residual = 0

        for layer in self.layers:
            hidden_state = layer(hidden_state)

            if isinstance(layer, ResidualMarker):
                hidden_state = residual + hidden_state
                residual = hidden_state

        return hidden_state

class HybridModel(BackboneModel):

    def __init__(self, module_names: list[str], x_dim: int, n_positions: int, n_embd: int = 128, y_dim: int = 1, **kwargs):
        backbone = HybridBackbone(
            module_names, 
            embed_dim=n_embd,
            n_positions=n_positions,
            **kwargs
        )
        super().__init__(backbone, x_dim, n_positions, n_embd, y_dim)

