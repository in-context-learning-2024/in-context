import torch
import torch.distributions as D

from typing import Any

from utils import RandomPermutationDistribution
from core import FunctionClass

class SparseParityRegression(FunctionClass):

    def __init__(self, k: int = 2, *args: Any, **kwargs: Any):
        self.k = k
        super(SparseParityRegression, self).__init__(*args, **kwargs)

    def _init_param_dist(self) -> D.Distribution:
        return RandomPermutationDistribution(self.k, self.x_dim)

    def evaluate(self, x_batch: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        indicies, *_ = params
        assert x_batch.shape[2] >= len(indicies)

        selected_x = x_batch[:, :, indices]

        y_batch = torch.prod(selected_x, dim=2)

        return y_batch.unsqueeze(-1)
