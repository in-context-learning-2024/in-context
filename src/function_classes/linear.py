import torch
import torch.distributions as D

from core import FunctionClass

class LinearRegression(FunctionClass):

    def _init_param_dist(self) -> D.Distribution:
        """Produce the distribution with which to sample parameters"""
        batch_shape = self.x_dist.batch_shape[:2]
        param_event_shape = torch.Size([self.y_dim, self.x_dim])

        param_dist_shape = torch.Size(batch_shape + param_event_shape)

        param_dist = D.Normal( torch.zeros(param_dist_shape), 
                               torch.ones(param_dist_shape)   )

        return param_dist

    def evaluate(self, x_batch: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        weights, *_ = params
        # TODO: changed this line so it would run, but it's probably wrong!
        partial_sums = torch.bmm(weights.squeeze(-2), x_batch.permute(0, 2, 1))
        full_sums = torch.sum(partial_sums, dim=-2, keepdim=True)
        y_batch = full_sums.squeeze()

        return y_batch

class SparseLinearRegression(LinearRegression):

    def __init__(self, sparsity: int = 3, *args, **kwargs):
        super(SparseLinearRegression, self).__init__(*args, **kwargs)
        self._sparsity = sparsity

    def evaluate(self, x_batch: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        weights, *_ = params

        param_shape = self.p_dist.batch_shape + self.p_dist.event_shape
        assert param_shape == weights.shape
        mask = torch.ones(param_shape).bool()

        for batch_entry in range(param_shape[0]):
            dimensions_to_keep: torch.Tensor = torch.randperm(self.x_dim)[:self._sparsity]
            mask[batch_entry, ..., dimensions_to_keep] = False

        weights[mask] = 0

        return super().evaluate(x_batch, weights)

class LinearClassification(LinearRegression):

    def evaluate(self, x_batch: torch.Tensor, *params: torch.Tensor):
        y_batch = super().evaluate(x_batch, *params)
        return y_batch.sign()

class QuadraticRegression(LinearRegression):

    def evaluate(self, x_batch: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        return super().evaluate(x_batch**2, *params)
