# pyright: reportIncompatibleMethodOverride=information
import torch.distributions as dist
import torch
import math

from typing import List, Optional, Any

def throw(ex):
    raise ex

def curried_throw(ex):
    return lambda *_, **__: throw(ex)


class CombinedDistribution(dist.Distribution):
    """Combine a number of unrelated distributions. i.e. combine a list of distributions to sample from in a combined call"""

    def __init__(self, *distributions: dist.Distribution):
        self._dists = distributions

        batch_shape = self._dists[0].batch_shape
        for dist in self._dists:
            assert dist.batch_shape == batch_shape, \
                f"Batch shapes do not match between all distributions! {batch_shape} != {dist.batch_shape} for {dist}"


    @property
    def batch_shape(self) -> torch.Size:
        return self._dists[0].batch_shape

    @property
    def event_shape(self) -> List[torch.Size]:
        return [ dist.event_shape for dist in self._dists ]

    def entropy(self) -> List[torch.Tensor]:
        return [ dist.entropy() for dist in self._dists ]

    def log_prob(self, value: torch.Tensor) -> List[torch.Tensor]:
        return [ dist.log_prob(value) for dist in self._dists ]

    @property
    def mean(self) -> List[torch.Tensor]:
        return [ dist.mean for dist in self._dists ]

    @property
    def mode(self) -> List[torch.Tensor]:
        return [ dist.mode for dist in self._dists ]

    def perplexity(self) -> List[torch.Tensor]:
        return [ dist.perplexity() for dist in self._dists ]

    def rsample(self, sample_shape: torch.Size) -> List[torch.Tensor]:
        return [ dist.rsample(sample_shape) for dist in self._dists ]

    def sample(self, sample_shape: torch.Size = torch.Size([])) -> List[torch.Tensor]:
        return [ dist.sample(sample_shape) for dist in self._dists ]

    def sample_n(self, n: int) -> List[torch.Tensor]:
        return [ dist.sample_n(n) for dist in self._dists ]

    @property
    def stddev(self) -> List[torch.Tensor]:
        return [ dist.stddev for dist in self._dists ]

    @property
    def support(self) -> List[Optional[Any]]:
        return [ dist.support for dist in self._dists ]

    @property
    def variance(self) -> List[torch.Tensor]:
        return [ dist.variance for dist in self._dists ]
    
class RandomPermutationDistribution(dist.Distribution):
    """A distribution that samples uniformly at random from the set of permutations of length k of {1, 2, ..., N}."""

    def __init__(self, k: int, N: int):
        super(RandomPermutationDistribution, self).__init__(validate_args=False)
        self.k = k
        self.N = N

    def sample(self, sample_shape: torch.Size = torch.Size()):
        return torch.randperm(self.N)[:self.k]
    
class SparseDistribution(dist.Distribution):
    """A distribution that returns xs sampled from {-1, 1} uniformly at random."""

    def __init__(self, batch_shape: torch.Size, event_shape: torch.Size, *args, **kwargs):
        super(SparseDistribution, self).__init__(*args, **kwargs | {"validate_args": False})
        self.batch_size = batch_shape[0]
        self.seq_len = batch_shape[1]
        self.x_dim = event_shape[0]

    def sample(self, sample_shape: torch.Size = torch.Size()):
        return 2 * torch.randint(0, 2, (self.batch_size, self.seq_len, self.x_dim)).float() - 1
    
    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size([self.batch_size, self.seq_len])

    @property
    def event_shape(self) -> torch.Size:
        return torch.Size([self.x_dim])
    
class RetrievalDistribution(dist.Distribution):
    """"""

    def __init__(self, batch_shape: torch.Size, event_shape: torch.Size, *args, **kwargs):
        super(RetrievalDistribution, self).__init__(*args, **kwargs | {"validate_args": False})
        self.batch_size = batch_shape[0]
        self.seq_len = batch_shape[1]
        self.x_dim = event_shape[0]

    def sample(self, query_pos: int = -1, sample_shape: torch.Size = torch.Size()):
        is_random_query = query_pos < 0
        x_batch = torch.randn(self.batch_size, 2 * self.seq_len + 1, self.x_dim)
        for i in range(self.batch_size):
            if is_random_query:
                query_pos = 2 * int(torch.randint(0, self.seq_len, torch.Size()))
            x_batch[i][-1, :] = x_batch[i][query_pos, :]
        x_batch = x_batch / torch.norm(x_batch, p=2, dim=-1, keepdim=True) * math.sqrt(self.x_dim)
        return x_batch

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size([self.batch_size, self.seq_len])

    @property
    def event_shape(self) -> torch.Size:
        return torch.Size([self.x_dim])
