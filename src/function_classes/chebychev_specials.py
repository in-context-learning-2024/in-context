import torch
import torch.distributions as D

from core import FunctionClass

class ChebychevSharedRoots(FunctionClass):

    def __init__(self, degree, perturbation=0.1, scaling_perc=0.2, *args, **kwargs):

        self.perturbation = perturbation
        self.scaling_perc = scaling_perc
        self._one_minus_one = torch.tensor([-1, 1])

        k = torch.arange(1, degree + 1)
        self.chebychev_roots = torch.cos((2 * k - 1) * torch.pi / (2 * degree)).view(1, -1)
        self.chebychev_roots = self.chebychev_roots.expand(kwargs['x_distribution'].batch_shape[0], -1)

        super(ChebychevSharedRoots, self).__init__(*args, **kwargs)

    def _init_param_dist(self) -> D.Distribution:
        """Produce the distribution with which to sample parameters"""
        perturbationd_dist = D.Uniform(-self.perturbation*torch.ones_like(self.chebychev_roots), self.perturbation*torch.ones_like(self.chebychev_roots))

        return perturbationd_dist
    
    def evaluate(self, x_batch: torch.Tensor, params: torch.Tensor) -> torch.Tensor:

        # Inside each batch, every x_point should be subtracted from each root
        roots = self.chebychev_roots + params
        # (batch_size, x_points, different roots)
        roots = roots.unsqueeze(1).expand(-1, x_batch.shape[1], -1)
        # (batch_size, different_points, repeated x_points)
        x_batch = x_batch.expand(-1, -1, roots.shape[-1])

        # Get values
        poly_values = torch.prod(x_batch - roots, dim=2)

        # Add some randomness to sign, and partially random scaling
        max_per_sample = torch.max(torch.abs(poly_values), dim=1).values
        poly_values = poly_values * self._one_minus_one[torch.randint(0, 2, (self.batch_size, 1))] * (self.scaling_perc * torch.rand((self.batch_size, 1)) + (1-self.scaling_perc))
        poly_values = poly_values / max_per_sample.unsqueeze(1)

        return poly_values