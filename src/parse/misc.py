import torch

from typing import Callable, Optional, Any

from core import TrainableModel

from .utils import (
    check_kwargs,
    clean_instantiate,
    YamlMap
)

def get_loss_fn(data: YamlMap) -> torch.nn.Module:
    
    LOSS_FNS = {
        "squared" : torch.nn.MSELoss,
        "mse" : torch.nn.MSELoss
    }

    check_kwargs(LOSS_FNS, data, "loss function")

    loss_fn_type: type[torch.nn.Module] | Callable[[], None] = LOSS_FNS[data['type']]
    
    return clean_instantiate(loss_fn_type, **data)

def get_optimizer(
        model: TrainableModel,
        data: YamlMap,
        optim_state: Optional[dict[Any, Any]] = None
    ) -> torch.optim.Optimizer:

    OPTIMIZERS = {
        "sgd" : torch.optim.SGD,
        "adam": torch.optim.Adam
    }

    check_kwargs(OPTIMIZERS, data, "optimizer")

    optim_type: type[torch.optim.Optimizer] = OPTIMIZERS[data['type']]

    opt = clean_instantiate(optim_type, model.parameters(), **data)

    if optim_state is not None:
        opt.load_state_dict(optim_state)
    
    return opt
