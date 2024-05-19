import torch
import torch.distributions as D

from typing import Any

from utils import SparseDistribution
from .utils import check_kwargs, clean_instantiate

def get_distribution(batch_shape: torch.Size, event_shape: torch.Size, init_kwargs: dict[str, Any]) -> D.Distribution:
    full_shape = batch_shape + event_shape

    DISTRIBUTION_BANK: dict[str, tuple[type[D.Distribution], dict[str, Any]]] = {
        "normal" : (
            D.MultivariateNormal,
            { "loc" : torch.zeros(full_shape),
              "covariance_matrix" : torch.eye(event_shape[-1]),
            }
        ),
        "uniform" : (
            D.Uniform,
            { "low"  : -torch.ones(full_shape),
              "high" : torch.ones(full_shape),
            }
        ),
        "sparse": (
            SparseDistribution,
            { "batch_shape" : batch_shape,
              "event_shape" : event_shape,
            }
        ),
    }

    valid_dists = {
        alias : class_type
        for alias, (class_type, _) 
        in DISTRIBUTION_BANK.items()
    }

    check_kwargs(valid_dists, init_kwargs, "Distribution")

    dist_info: tuple[type[D.Distribution], dict[str, Any]] = DISTRIBUTION_BANK[init_kwargs['type']]
    dist_class, kwargs = dist_info

    del init_kwargs['type']
    kwargs.update(init_kwargs)
    kwargs.update({ "validate_args" : True })

    return clean_instantiate(dist_class, **kwargs)

def get_x_distribution(batch_size: int, seq_len: int, x_dim: int, init_kwargs: dict[str, Any]) -> D.Distribution:
    batch_shape = torch.Size([batch_size, seq_len])
    event_shape = torch.Size([x_dim, ])
    
    return get_distribution(batch_shape, event_shape, init_kwargs)
