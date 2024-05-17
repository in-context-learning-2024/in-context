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
        
        y_batch = torch.ones(x_batch.shape[0], x_batch.shape[1], device=x_batch.device)
        for idx in indicies:
            y_batch *= x_batch[:, :, idx]

        return y_batch.unsqueeze(-1)
