from typing import Any

from ..transformer import BackboneModel
from .backbone import HybridBackbone

class HybridModel(BackboneModel):

    def __init__(self,
            module_names: list[str],
            x_dim: int,
            n_positions: int,
            n_embd: int = 128,
            y_dim: int = 1,
            **kwargs: Any
        ):
        backbone = HybridBackbone(
            module_names, 
            embed_dim=n_embd,
            n_positions=n_positions,
            **kwargs
        )
        super().__init__(backbone, x_dim, n_positions, n_embd, y_dim)