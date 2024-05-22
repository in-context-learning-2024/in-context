from transformers import (
    GPT2Config, 
    GPT2Model, 
    LlamaConfig, 
    LlamaModel,
    MambaConfig,
    MambaModel,
 ) # pyright: ignore[reportPrivateImportUsage]

from torch import nn, Tensor
from typing import Any

from core import TrainableModel

class BackboneModel(TrainableModel):

    def __init__(self, 
            backbone: nn.Module, 
            x_dim: int, 
            n_positions: int, 
            n_embd: int=128, 
            y_dim: int = 1,
        ):
        super().__init__(x_dim, y_dim)

        self.context_length = n_positions
        self._read_in = nn.Linear(x_dim, n_embd)
        self._backbone = backbone
        self._read_out = nn.Linear(n_embd, y_dim)

    def forward(self, xs: Tensor, ys: Tensor):
        self._backbone.to(xs.device) # pyright: ignore[reportArgumentType,reportAttributeAccessIssue]

        zs = self.interleave(xs, ys)

        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state # pyright: ignore[reportCallIssue]
        prediction = self._read_out(output)

        return prediction[:, ::2] # predict only on xs

class GPT2(BackboneModel):

    def __init__(self,
            x_dim: int,
            n_positions: int,
            n_embd: int = 128,
            n_layer: int = 12,
            n_head: int = 4,
            y_dim: int = 1,
            **kwargs: Any
        ):

        configuration = GPT2Config(
            # vocab_size=1,
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )

        self.gpt2_configuration = configuration
        backbone: nn.Module = GPT2Model(configuration) # pyright: ignore[reportAssignmentType]

        super().__init__(backbone, x_dim, n_positions, n_embd, y_dim=y_dim,)

        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"


class Llama(BackboneModel):

    def __init__(self,
            x_dim: int,
            n_positions: int,
            n_embd: int = 128,
            n_layer: int = 12,
            n_head: int = 4,
            hidden_act: str = 'silu',
            rope_theta: float = 1e4,
            y_dim: int = 1,
            **kwargs: Any
        ):

        configuration = LlamaConfig(
            vocab_size=1,
            max_position_embeddings=2 * n_positions,
            hidden_size=n_embd,
            intermediate_size=4*n_embd,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            hidden_act=hidden_act,
            rope_theta=rope_theta,
            use_cache=False,
        )

        self.llama_configuration = configuration
        backbone: nn.Module = LlamaModel(configuration) # pyright: ignore[reportAssignmentType]

        super().__init__(backbone, x_dim, n_positions, n_embd, y_dim=y_dim,)

        self.name = f"llama_embd={n_embd}_layer={n_layer}_head={n_head}"

class Mamba(BackboneModel):

    def __init__(self,
            x_dim: int,
            n_positions: int,
            n_embd: int = 128,
            n_layer: int = 12,
            y_dim: int = 1,
            **kwargs: Any
        ):

        configuration = MambaConfig(
            vocab_size=1,
            hidden_size=n_embd,
            state_size=16,
            expand=4,
            num_hidden_layers=n_layer,
            use_cache=False,
        )

        self.mamba_configuration = configuration
        backbone: nn.Module = MambaModel(configuration)

        super().__init__(backbone, x_dim, n_positions, n_embd, y_dim=y_dim,)

        self.name = f"mamba_embd={n_embd}_layer={n_layer}"
