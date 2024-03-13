import torch
from torch.distributions.distribution import Distribution
from typing import Tuple

class FunctionClass:

    def __init__(self, x_distribution: Distribution, param_distribution_class: type[Distribution], y_dim: int = 1):

        # we should pull as much information from the `x_distribution` if 
        #   we expect the instatiating code to provide it
        #   torch.(...).Distribution docs: https://pytorch.org/docs/stable/distributions.html

        assert len(x_distribution.event_shape) == 1, f"Found more than one event axis for input distribution. Inputs must be vectors!"

        self.batch_size = x_distribution.batch_shape[0]
        self.sequence_length = x_distribution.batch_shape[1]
        self.x_dim = x_distribution.event_shape[0]
        self.y_dim = y_dim

        self._x_dist = x_distribution
        self._p_dist = param_distribution_class(
            batch_shape = torch.Size([self.batch_size]),
            event_shape = self._get_parameter_shape(self.x_dim, self.y_dim)
        )

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x_batch = self._x_dist.sample()
        y_batch = self.evaluate(x_batch)
        return x_batch, y_batch

    @staticmethod
    def _get_parameter_shape(x_dim: int, y_dim: int=1) -> torch.Size:
        raise NotImplementedError(f"Abstract class FunctionClass does not have a parameter shape!")

    def evaluate(self, x_batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"Abstract class FunctionClass does not implement `.evaluate(xs)`!")

"""
Perhaps some pseudo-implementation-code is in order:

for i, (x_batch, y_batch) in zip(range(training_steps), MyFunctionClass(*my_args, **my_kwargs)):
    
    ys_pred = model.predict(x_batch, y_batch)
    loss = loss_fn(ys_pred, y_batch)

    loss.backward()
"""