
import torch.distributions as dist
import torch

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
        # raise TypeError(f"Combined distributions do not have an event_shape!")
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
    
    def sample(self, sample_shape: torch.Size) -> List[torch.Tensor]:
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
