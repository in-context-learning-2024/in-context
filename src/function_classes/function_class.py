import torch
from torch.distributions.distribution import Distribution
from typing import Tuple, List, Optional

class FunctionClass:

    def __init__(self, x_distribution: Distribution, param_distribution_class: type[Distribution], x_curriculum_dim: Optional[int] = None, y_dim: int = 1):

        # we should pull as much information from the `x_distribution` if 
        #   we expect the instatiating code to provide it
        #   torch.(...).Distribution docs: https://pytorch.org/docs/stable/distributions.html

        assert len(x_distribution.event_shape) == 1, f"Found more than one event axis for input distribution. Inputs must be vectors!"

        self.batch_size = x_distribution.batch_shape[0]
        self.sequence_length = x_distribution.batch_shape[1]
        self.x_dim = x_distribution.event_shape[0]
        self.y_dim = y_dim

        self.x_curriculum_dim = x_curriculum_dim or self.x_dim

        self._x_dist = x_distribution
        self._p_dist = self._get_param_dist(param_distribution_class)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x_batch_tmp: torch.Tensor = self._x_dist.sample()
        x_batch = torch.zeros_like(x_batch_tmp)
        x_batch[..., :self.x_curriculum_dim] = x_batch_tmp[..., :self.x_curriculum_dim]

        params: List[torch.Tensor] | torch.Tensor  = self._p_dist.sample()
        y_batch: torch.Tensor = self.evaluate(x_batch, params)
        return x_batch, y_batch

    def _get_param_dist(self, param_distribution_class: type[Distribution]) -> Distribution:
        """Produce the distribution with which to sample parameters using properties of this instance"""
        return param_distribution_class(
            batch_shape = torch.Size([self.batch_size]),
            event_shape = self._parameter_shape
        )

    @property
    def _parameter_shape(self) -> torch.Size:
        """Produce the shape of a single sample of the parameters for a function from this function class. Supporting property for `_get_param_dist(...)`"""
        raise NotImplementedError(f"Abstract class FunctionClass does not have a parameter shape!")

    def evaluate(self, x_batch: torch.Tensor, params: List[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Produce a Tensor of shape (batch_size, sequence_length, y_dim) given a Tensor of shape (batch_size, sequence_length, x_dim)"""
        raise NotImplementedError(f"Abstract class FunctionClass does not implement `.evaluate(xs)`!")

"""
Perhaps some pseudo-implementation-code is in order:

for i, (x_batch, y_batch) in zip(range(training_steps), MyFunctionClass(*my_args, **my_kwargs)):
    
    ys_pred = model.predict(x_batch, y_batch)
    loss = loss_fn(ys_pred, y_batch)

    loss.backward()
"""