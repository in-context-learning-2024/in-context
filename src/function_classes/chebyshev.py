import torch
import torch.distributions as D

from torch import Tensor
from typing import Any

from core import FunctionClass

def generate_chebyshev_coefficients(lowest_degree: int, highest_degree: int) -> Tensor:
    # Create a matrix to hold the coefficients, initializing with zeros
    n = highest_degree + 1
    coeffs = torch.zeros(n, n, dtype=torch.int32)
    
    # Initial conditions for T_0(x) and T_1(x)
    coeffs[0, 0] = 1  # T_0(x) = 1
    coeffs[1, 1] = 1  # T_1(x) = x
    
    # Use the recurrence relation T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)
    skew_index = torch.arange(1, n, dtype=torch.long)
    for i in range(2, n):
        coeffs[i , :][skew_index] = 2 * coeffs[i-1, :][:-1]
        coeffs[i , :] -= coeffs[i-2, :]
    
    return coeffs[lowest_degree:highest_degree+1]

class ChebyshevKernelLinearRegression(FunctionClass):

    """
    Class for generating linear combinations of Chebyshev polynomials in given interval of degrees
    Linear combinations are generated randomly by sampling from a normal distribution
    """

    def __init__(self, lowest_degree: int = 3, highest_degree: int = 11, *args: Any, **kwargs: Any):

        self.chebyshev_coeffs = generate_chebyshev_coefficients(lowest_degree, highest_degree).float()
        self.lowest_degree = lowest_degree
        self.highest_degree = highest_degree

        super(ChebyshevKernelLinearRegression, self).__init__(*args, **kwargs)

    def _init_param_dist(self) -> D.Distribution:
        """Produce the distribution with which to sample parameters"""
        # Combine each polynomial randomly per sample
        # combinations: (batch_size, 1 coefficient for each poly in chebyshev_coeffs, seq_length)
        combinations_dist = D.Normal(torch.zeros((self.batch_size, 1, self.chebyshev_coeffs.shape[0])), 1)
        return combinations_dist


    def evaluate(self, x_batch: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:

        combinations, *_ = params

        # Generate x to the power of i
        # x_pows: (batch_size, seq_length, repeated x_values but to the power of different degrees)
        x_pows = x_batch.pow(torch.arange(0, self.highest_degree + 1))

        # chebyshev_coeffs: (different basis polys, one element for each possible degree of poly)
        # basis_polys: (batch_size, 1 value for each poly in chebyshev_coeffs, seq_length)
        basis_polys = self.chebyshev_coeffs @ x_pows.permute(0, 2, 1)

        # Only include coefficients up to random degree
        indices = torch.arange(0, self.chebyshev_coeffs.shape[0]).unsqueeze(0).expand(self.batch_size, -1)
        rand_tresh = torch.randint(0, self.chebyshev_coeffs.shape[0], (self.batch_size, 1))
        mask_indices = (rand_tresh < indices).unsqueeze(1)
        combinations[mask_indices] = 0

        # Combine basis polynomials into 1
        return (combinations @ basis_polys).squeeze(1)

class ChebyshevSharedRoots(FunctionClass):

    """
    This class generates chebyshev polynomials with shared roots
    Roots can be uniformly randomly perturbed
    """

    def __init__(self, degree: int = 5, perturbation: float = 0.1, *args: Any, **kwargs: Any):

        self.perturbation = perturbation
        self._one_minus_one = torch.tensor([-1, 1])

        k = torch.arange(1, degree + 1)
        self.chebyshev_roots = torch.cos((2 * k - 1) * torch.pi / (2 * degree)).view(1, -1)
        self.chebyshev_roots = self.chebyshev_roots.expand(kwargs['x_distribution'].batch_shape[0], -1)

        super(ChebyshevSharedRoots, self).__init__(*args, **kwargs)

    def _init_param_dist(self) -> D.Distribution:
        """Produce the distribution with which to sample parameters"""
        perturbationd_dist = D.Uniform(-self.perturbation*torch.ones_like(self.chebyshev_roots), self.perturbation*torch.ones_like(self.chebyshev_roots))
        return perturbationd_dist
    
    def evaluate(self, x_batch: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:

        perturbations, *_ = params

        # Inside each batch, every x_point should be subtracted from each root
        roots = self.chebyshev_roots + perturbations
        # (batch_size, x_points, different roots)
        roots = roots.unsqueeze(1).expand(-1, x_batch.shape[1], -1)
        # (batch_size, different_points, repeated x_points)
        x_batch = x_batch.expand(-1, -1, roots.shape[-1])

        # Multiply factors together to get polynomial values
        poly_values = torch.prod(x_batch - roots, dim=2)

        # Add some randomness to sign, and partially random scaling
        max_per_sample = torch.max(torch.abs(poly_values), dim=1).values
        poly_values = poly_values * self._one_minus_one[torch.randint(0, 2, (self.batch_size, 1))] / max_per_sample.unsqueeze(1)

        return poly_values
