import inspect

from typing import Callable, Any
from importlib import import_module

import torch

from torch import nn, Tensor

from transformers import PretrainedConfig

from transformers.models.mamba.modeling_mamba import (
    MambaMixer,
    MambaBlock,
)

from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaAttention,
    LlamaRotaryEmbedding,
    LlamaMLP,
    LlamaDecoderLayer,
)

from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2MLP,
    GPT2Block,
)


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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.Identity()
    
    def forward(self, *args: Any, **kwargs: Any):
        return self.net(*args, **kwargs)


class AbsolutePositionalEmbedding(nn.Module):

    def __init__(self, max_num_positions: int, hidden_dim: int):
        super().__init__()
        self.positions = torch.arange(0, max_num_positions, dtype=torch.int)
        self.embed = nn.Embedding(max_num_positions, hidden_dim)

    def forward(self, inp: Tensor) -> Tensor:
        *_, seq_len, _ = inp.shape
        pos = self.positions.to(device=inp.device)
        embeddings = self.embed(pos)[:seq_len]
        return inp + embeddings

class _RotaryEmbeddingStub(LlamaRotaryEmbedding):
    def __init__(self, *args: Any, enable: bool = False, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.enable = enable

    def forward(self, x: Tensor, position_ids: Tensor):
        cos, sin = super().forward(x, position_ids)

        if self.enable:
            return cos, sin
        return torch.ones_like(cos), torch.zeros_like(sin)

def _make_llama_attention_factory(config: PretrainedConfig, layer_idx: int, use_rope: bool) -> Callable[[], nn.Module]:
    def llama_attn_factory():
        attn_module = LlamaAttention(config=config, layer_idx=layer_idx) # pyright: ignore[reportArgumentType]

        attn_module.rotary_emb = _RotaryEmbeddingStub(
            config=config,
            device=None,
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
        "mamba mixer" : lambda: MambaMixer(config, layer_idx=layer_idx), # pyright: ignore[reportArgumentType]

        "llama mlp" : lambda: LlamaMLP(config),
        "gpt2 mlp"  : lambda: GPT2MLP(config.n_inner if config.n_inner is not None else 4 * config.hidden_size, config),

        "llama block" : lambda: LlamaDecoderLayer(config, layer_idx), # pyright: ignore[reportArgumentType]
        "gpt2 block"  : lambda: GPT2Block(config, layer_idx=layer_idx),
        "mamba block" : lambda: MambaBlock(config, layer_idx=layer_idx),
    }


    if not all([key in SUPPORTED_BLOCKS for key in MAPPING.keys()]):
        raise Exception("Not all \"supported\" blocks can be instantiated! Make "
                        "sure `MAPPING` in this function and `SUPPORTED_BLOCKS` "
                        "are equal in this file")
    
    if spec_name in MAPPING:
        return MAPPING[spec_name]()

    # if this isn't a predefined module, do some dynamic magic
    ## dynamically import the class
    lib_name, class_name = spec_name.split("#")
    lib = import_module(lib_name)
    module_class = eval(f"lib.{class_name}")
    if not inspect.isclass(module_class):
        raise TypeError(f"Could not instantiate hybrid submodule!: {module_class} from {lib_name} is not a class!")

    ## instantiate the class with config (and optionally layer_idx)
    args: dict[str, Any] = { "config" : config }
    sig = inspect.signature(module_class)
    if "layer_idx" in sig.parameters:
        args["layer_idx"] = layer_idx
    
    return module_class(**args)

    

