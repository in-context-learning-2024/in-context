import torch
from torch.distributions.distribution import Distribution

from function_classes.function_class import FunctionClass

class LinearRegression(FunctionClass):

    def __init__(self, x_distribution: Distribution, param_distribution_class: type[Distribution]):
        super(LinearRegression, self).__init__(x_distribution, param_distribution_class)

    @staticmethod
    def _get_parameter_shape(x_dim: int, y_dim: int=1) -> torch.Size:
        return torch.Size([y_dim, x_dim])

    def evaluate(self, x_batch: torch.Tensor) -> torch.Tensor:
        params = self._p_dist.sample().to(x_batch.device)

        partial_sums = torch.bmm(params, x_batch.unsqueeze(-1))
        full_sums = torch.sum(partial_sums, dim=-2, keepdim=True)
        y_batch = full_sums.squeeze()

        return y_batch

class SparseLinearRegression(LinearRegression):

    def __init__(self, 
        x_distribution: Distribution, 
        param_distribution_class: type[Distribution],
        sparsity: int = 3
    ):
        super(SparseLinearRegression, self).__init__(x_distribution, param_distribution_class)
        self._sparsity = sparsity

    def evaluate(self, x_batch: torch.Tensor) -> torch.Tensor:
        params = self._p_dist.sample().to(x_batch.device)

        param_shape = self._get_parameter_shape(self.x_dim, self.y_dim)
        mask = torch.ones(param_shape).bool()
        mask[torch.randperm(self.x_dim)[:self._sparsity]] = False
        params[:, mask] = 0

        partial_sums = torch.bmm(params, x_batch.unsqueeze(-1))
        full_sums = torch.sum(partial_sums, dim=-2, keepdim=True)
        y_batch = full_sums.squeeze()

        return y_batch

class LinearClassification(LinearRegression):
    def __init__(self, x_distribution: Distribution, param_distribution_class: type[Distribution]):
        super().__init__(x_distribution, param_distribution_class)

    def evaluate(self, x_batch: torch.Tensor):
        y_batch = super().evaluate(x_batch)
        return y_batch.sign()

class QuadraticRegression(LinearRegression):

    def __init__(self, x_distribution: Distribution, param_distribution_class: type[Distribution]):
        super(QuadraticRegression, self).__init__(x_distribution, param_distribution_class)

    def evaluate(self, x_batch: torch.Tensor) -> torch.Tensor:
        params = self._p_dist.sample().to(x_batch.device)

        partial_sums = torch.bmm(params, (x_batch**2).unsqueeze(-1))
        full_sums = torch.sum(partial_sums, dim=-2, keepdim=True)

        # Renormalize to Linear Regression Scale
        full_sums /= torch.sqrt(torch.Tensor(3))

        y_batch = full_sums.squeeze()

        return y_batch
