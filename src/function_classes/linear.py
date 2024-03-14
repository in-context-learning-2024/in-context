import torch
from torch.distributions.distribution import Distribution

from function_classes.function_class import FunctionClass

class LinearRegression(FunctionClass):

    def __init__(self, *args, **kwargs):
        super(LinearRegression, self).__init__(*args, **kwargs)

    @property
    def _parameter_shape(self) -> torch.Size:
        return torch.Size([self.y_dim, self.x_dim])

    def evaluate(self, x_batch: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        partial_sums = torch.bmm(params, x_batch.unsqueeze(-1))
        full_sums = torch.sum(partial_sums, dim=-2, keepdim=True)
        y_batch = full_sums.squeeze()

        return y_batch

class SparseLinearRegression(LinearRegression):

    def __init__(self, sparsity: int = 3, *args, **kwargs):
        super(SparseLinearRegression, self).__init__(*args, **kwargs)
        self._sparsity = sparsity

    def evaluate(self, x_batch: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        param_shape = self._parameter_shape
        mask = torch.ones(param_shape).bool()
        mask[torch.randperm(self.x_dim)[:self._sparsity]] = False
        params[:, mask] = 0

        partial_sums = torch.bmm(params, x_batch.unsqueeze(-1))
        full_sums = torch.sum(partial_sums, dim=-2, keepdim=True)
        y_batch = full_sums.squeeze()

        return y_batch

class LinearClassification(LinearRegression):
    def __init__(self, *args, **kwargs):
        super(LinearClassification, self).__init__(*args, **kwargs)

    def evaluate(self, x_batch: torch.Tensor, params: torch.Tensor):
        y_batch = super().evaluate(x_batch, params)
        return y_batch.sign()

class QuadraticRegression(LinearRegression):

    def __init__(self, *args, **kwargs):
        super(QuadraticRegression, self).__init__(*args, **kwargs)

    def evaluate(self, x_batch: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        partial_sums = torch.bmm(params, (x_batch**2).unsqueeze(-1))
        full_sums = torch.sum(partial_sums, dim=-2, keepdim=True)

        # Renormalize to Linear Regression Scale
        full_sums /= torch.sqrt(torch.Tensor(3))

        y_batch = full_sums.squeeze()

        return y_batch
