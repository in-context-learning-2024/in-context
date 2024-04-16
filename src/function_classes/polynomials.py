import torch
import torch.distributions as D
import numpy as np
import random

from core import FunctionClass

class PolynomialMixedSliced(FunctionClass):

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
        
        self.mask = torch.ones((highest_degree - lowest_degree + 1, highest_degree), dtype=torch.bool)
        for i, degree in enumerate(range(lowest_degree, highest_degree + 1)):
            self.mask[i][:degree] = 0

        super(PolynomialMixedSliced, self).__init__(*args, **kwargs)

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
        roots = 2*torch.rand((x_batch.shape[0], 1, self.highest_degree)) - 1
        perturb = 2*self.perturbation * torch.rand((x_batch.shape[0], len(idx_slices), self.highest_degree)) - self.perturbation
        roots = roots + perturb

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