import torch
import torch.distributions as D

from typing import Any

from utils import RandomMaskDistribution
from core import FunctionClass

class SparseParityRegression(FunctionClass):

    def __init__(self, k: int = 2, *args: Any, **kwargs: Any):
        self.k = k
        super(SparseParityRegression, self).__init__(*args, **kwargs)

    def _init_param_dist(self) -> D.Distribution:
        return RandomMaskDistribution(self.k, self.x_dim, self.batch_size)

    def evaluate(self, x_batch: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        masks, *_ = params

        # mask out xs with 1s
        selected_x = x_batch * masks.unsqueeze(1) 
        selected_x += 1 - masks.unsqueeze(1) 

        # multiply all xs to see the parity of non-masked out xs
        y_batch = torch.prod(selected_x, dim=2)

        return y_batch.unsqueeze(-1)
