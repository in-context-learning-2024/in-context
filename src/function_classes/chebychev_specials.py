import torch
import torch.distributions as D
import numpy as np
import random

from core import FunctionClass

class ChebychevSharedRoots(FunctionClass):

    """
    This class generates polynomials with shared roots
    Roots can be perturbed
    And the polynomials are mainly scaled so that the maximum value is 1
    Although scaling_perc is the percentage of scaling that is uniform random
    """

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
    

class ChebychevSliced(FunctionClass):

    """
    Output is a sliced polynomial, concatenated at random intervals.
    Degree of the polynomial is random per sample
    Lets you learn about the function globally, while also learning about the local structure of the polynomials.

    min_slices: minimum number of slices (cuts) to make in the x-axis
    max_slices: maximum number of slices (cuts) to make in the x-axis
    lowest_degree: lowest degree of polynomials to generate
    highest_degree: highest degree of polynomials to generate
    perturbation: how much to perturb the roots
    scaling_perc: percentage of scaling to be random, remaining part of scaling is normalized
    """

    def __init__(self, min_slices: int = 1, max_slices: int = 5, lowest_degree: int = 3, highest_degree: int = 11, 
                 perturbation=0.3, scaling_perc=0.3, *args, **kwargs):

        self.min_slices = min_slices
        self.max_slices = max_slices
        self.batch_size = kwargs['x_distribution'].batch_shape[0]
        self.perturbation = perturbation

        self.lowest_degree = lowest_degree
        self.highest_degree = highest_degree

        self._one_minus_one = torch.tensor([-1, 1])
        self.scaling_perc = scaling_perc

        k = torch.arange(1, highest_degree + 1)
        self.chebychev_roots = torch.cos((2 * k - 1) * torch.pi / (2 * highest_degree))

        self.chebychev_roots = self.chebychev_roots.unsqueeze(0).expand(self.batch_size, -1)
        self.mask = torch.ones((highest_degree - lowest_degree + 1, highest_degree), dtype=torch.bool)
        for i, degree in enumerate(range(lowest_degree, highest_degree + 1)):
            self.mask[i][:degree] = 0

        super(ChebychevSliced, self).__init__(*args, **kwargs)
        
    def _init_param_dist(self) -> D.Distribution:
        """Produce the distribution with which to sample parameters"""
        perturbationd_dist = D.Uniform(-self.perturbation*torch.ones_like(self.chebychev_roots), self.perturbation*torch.ones_like(self.chebychev_roots))

        return perturbationd_dist

    def _random_interval_rearange_indexes(self, x_batch: torch.Tensor) -> torch.Tensor:

        # Choose a random number of slices for each sample in the batch
        slice_num = np.random.randint(self.min_slices, self.max_slices + 1, size=1)
        rand_idxs = tuple(np.sort(np.random.choice(x_batch.shape[1], size=slice_num, replace=False)))

        idx = torch.arange(0, x_batch.shape[1], 1)
        idx_slices = list(torch.tensor_split(idx, rand_idxs))
        random.shuffle(idx_slices)
        idx = torch.cat(idx_slices, dim=0)

        return idx

    def evaluate(self, x_batch: torch.Tensor, params: torch.Tensor) -> torch.Tensor:

        # Rearange xs_b
        idx = self._random_interval_rearange_indexes(x_batch)
        x_batch = x_batch[:, idx]

        # Perturb roots
        roots = self.chebychev_roots + params

        # (batch_size, x_points, different_roots)
        roots = roots.unsqueeze(1).expand(-1, x_batch.shape[1], -1)
        # (batch_size, different_points, repeated x_points)
        x_batch = x_batch.expand(-1, -1, roots.shape[-1])

        # Create mask according to given degrees
        degrees = torch.randint(low=0, high=self.highest_degree-self.lowest_degree+1, size=(self.batch_size,))
        current_mask = self.mask[degrees, :].unsqueeze(1).expand(-1, x_batch.shape[1], -1)

        # Perform subtraction + mask and multiplication
        vals = x_batch - roots
        vals[current_mask] = 1
        poly_values = torch.prod(vals, dim=2)

        # Add some randomness to sign, and partially random scaling
        max_per_sample = torch.max(torch.abs(poly_values), dim=1).values
        poly_values = poly_values * self._one_minus_one[torch.randint(0, 2, (self.batch_size, 1))] * (self.scaling_perc * torch.rand((self.batch_size, 1)) + (1-self.scaling_perc))
        poly_values = poly_values / max_per_sample.unsqueeze(1)

        return poly_values


class ChebychevMixedSliced(FunctionClass):
    """
    Creates a task where the output is several polynomials with shared roots, but concatenated at random intervals.
    Degree of the polynomial is random but shared per sample (sequence)
    Lets you learn about the roots globally, while also learning about the local structure of the polynomials.

    min_slices: minimum number of slices (cuts) to make in the x-axis
    max_slices: maximum number of slices (cuts) to make in the x-axis
    lowest_degree: lowest degree of polynomials to generate
    highest_degree: highest degree of polynomials to generate
    perturbation: how much to perturb the roots
    scaling_perc: percentage of scaling to be random, remaining part of scaling is normalized
    """

    def __init__(self, min_slices: int = 1, max_slices: int = 5, lowest_degree: int = 3, highest_degree: int = 11, 
                 perturbation=0.3, scaling_perc=0.3, *args, **kwargs):

        self.min_slices = min_slices
        self.max_slices = max_slices
        self.batch_size = kwargs['x_distribution'].batch_shape[0]
        self.perturbation = perturbation

        self.lowest_degree = lowest_degree
        self.highest_degree = highest_degree

        self._one_minus_one = torch.tensor([-1, 1])
        self.scaling_perc = scaling_perc

        k = torch.arange(1, highest_degree + 1)
        self.chebychev_roots = torch.cos((2 * k - 1) * torch.pi / (2 * highest_degree))

        self.chebychev_roots = self.chebychev_roots.unsqueeze(0).expand(self.batch_size, -1) # (batch_size, different_roots)
        self.mask = torch.ones((highest_degree - lowest_degree + 1, highest_degree), dtype=torch.bool)
        for i, degree in enumerate(range(lowest_degree, highest_degree + 1)):
            self.mask[i][:degree] = 0

        super(ChebychevMixedSliced, self).__init__(*args, **kwargs)

    def _init_param_dist(self) -> D.Distribution:
        """Produce the distribution with which to sample parameters"""
        return D.Uniform(0, 1)
        

    def _random_interval_rearange_indexes(self, x_batch: torch.Tensor) -> torch.Tensor:

        # Choose a random number of slices for each sample in the batch
        slice_num = np.random.randint(self.min_slices, self.max_slices + 1, size=1)
        rand_idxs = tuple(np.sort(np.random.choice(x_batch.shape[1], size=slice_num, replace=False)))

        idx = torch.arange(0, x_batch.shape[1], 1)
        idx_slices = list(torch.tensor_split(idx, rand_idxs))
        random.shuffle(idx_slices)

        return idx_slices

    def evaluate(self, x_batch: torch.Tensor, params: torch.Tensor) -> torch.Tensor:

        # Rearange xs_b
        idx_slices = self._random_interval_rearange_indexes(x_batch)

        # Create mask according to given degrees
        degrees = torch.randint(low=0, high=self.highest_degree-self.lowest_degree+1, size=(self.batch_size,))

        # Perturb roots
        # self.chebyshev_roots (batch_size, different_roots)
        # chebychev_roots (batch_size, slices, different_roots)
        chebychev_roots = self.chebychev_roots.unsqueeze(1).expand(-1, len(idx_slices), -1)
        roots = chebychev_roots + 2*self.perturbation*torch.rand((x_batch.shape[0], 1, chebychev_roots.shape[-1])) - self.perturbation

        # (batch_size, slices, x_points, different_roots)
        roots = roots.unsqueeze(2).expand(-1, -1, x_batch.shape[1], -1)
        # (batch_size, slices, different_points, repeated x_points)
        x_batch = x_batch.unsqueeze(1).expand(-1, len(idx_slices), -1, roots.shape[-1])

        # current_mask (batch_size, slices, different_points, roots)
        current_mask = self.mask[degrees, :].unsqueeze(1).unsqueeze(1).expand(-1, len(idx_slices), x_batch.shape[2], -1)

        # Perform subtraction + mask and multiplication
        vals = x_batch - roots
        vals[current_mask] = 1
        poly_values = torch.prod(vals, dim=3)

        # Add some randomness to sign, and partially random scaling
        poly_val_collection = []
        for i, idx_slice in enumerate(idx_slices):
            relevant_poly_values = poly_values[:, i, idx_slice]

            if relevant_poly_values.shape[1] != 0:
                max_val = torch.max(torch.abs(relevant_poly_values), dim=1).values
                relevant_poly_values = relevant_poly_values * self._one_minus_one[torch.randint(0, 2, (self.batch_size, 1))] * (self.scaling_perc * torch.rand((self.batch_size, 1)) + (1-self.scaling_perc))
                relevant_poly_values = relevant_poly_values / max_val.unsqueeze(1)
            
            poly_val_collection.append(relevant_poly_values)

        final_poly_values = torch.cat(poly_val_collection, dim=1)

        return final_poly_values