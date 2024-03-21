import torch
from torch.distributions.distribution import Distribution
from typing import Tuple, List, Optional

class FunctionClass:

    def __init__(
            self, 
            x_distribution: Distribution,
            x_curriculum_dim: Optional[int] = None, 
            y_dim: int = 1
        ):

        # we pull as much information from the `x_distribution` as possible, so all 
        #   torch.(...).Distribution docs: https://pytorch.org/docs/stable/distributions.html

        assert len(x_distribution.event_shape) == 1 or len(x_distribution.batch_shape) == 3, \
            f"Supplied x dimension is not 1. Inputs must be vectors!" + \
            f"Expected: [batch_size, seq_len, x_dim]" + \
            f"Got: {x_distribution.batch_shape + x_distribution.event_shape}"

        if len(x_distribution.event_shape) == 1:
            self.batch_size, self.sequence_length, *_ = x_distribution.batch_shape
            self.x_dim: int = x_distribution.event_shape[0]
        else:
            self.batch_size, self.sequence_length, *_, self.x_dim = x_distribution.batch_shape
        self.y_dim: int = y_dim

        # The number of dimensions to keep after sampling
        self.x_curriculum_dim: int = x_curriculum_dim or self.x_dim

        # The distribution with which to sample x vectors
        self.x_dist: Distribution = x_distribution

        # The distribution with which to sample parameters
        self.p_dist: Distribution = self._init_param_dist()

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x_batch_tmp: torch.Tensor = self.x_dist.sample()
        x_batch = torch.zeros_like(x_batch_tmp)
        x_batch[..., :self.x_curriculum_dim] = x_batch_tmp[..., :self.x_curriculum_dim]

        params: List[torch.Tensor] | torch.Tensor  = self.p_dist.sample()
        y_batch: torch.Tensor = self.evaluate(x_batch, params)
        return x_batch, y_batch

    def _init_param_dist(self) -> Distribution:
        """Produce the distribution with which to sample parameters"""
        raise NotImplementedError(f"Abstract class FunctionClass does not have a parameter distribution!")

    def evaluate(self, x_batch: torch.Tensor, params: List[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Produce a Tensor of shape (batch_size, sequence_length, y_dim) given a Tensor of shape (batch_size, sequence_length, x_dim)"""
        raise NotImplementedError(f"Abstract class FunctionClass does not implement `.evaluate(xs)`!")

class ModifiedFunctionClass(FunctionClass):

    def __init__(self, inner_function_class: FunctionClass):
        self._in_fc = inner_function_class

        self.x_dist = self._in_fc.x_dist
        self.p_dist = self._in_fc.p_dist

        self.batch_size = self._in_fc.batch_size
        self.sequence_length = self._in_fc.sequence_length
        self.x_dim = self._in_fc.x_dim
        self.x_curriculum_dim = self._in_fc.x_curriculum_dim
        self.y_dim = self._in_fc.y_dim
