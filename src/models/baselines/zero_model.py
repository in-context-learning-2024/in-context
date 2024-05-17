from torch import Tensor
from typing import Any

from core import Baseline

class ZeroModel(Baseline):
    def __init__(self, **kwargs: Any):
        super(ZeroModel, self).__init__(**kwargs)
        self.name = "zero_model"
        self.context_length = -1

    def evaluate(self, xs: Tensor, ys: Tensor):
        return 0 * xs[..., 0:self.y_dim]
