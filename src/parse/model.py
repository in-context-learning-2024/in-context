import torch

from typing import Optional

from core import ContextModel

from models import MODELS

from .utils import check_kwargs, clean_instantiate


def get_model(init_kwargs: dict, x_dim: int, y_dim: int, model_weights: Optional[dict] = None) -> ContextModel:

    check_kwargs(MODELS, init_kwargs, "model")
    
    model_class: type[ContextModel] = MODELS[init_kwargs['type']]

    init_kwargs |= { "x_dim" : x_dim, "y_dim" : y_dim }
    model = clean_instantiate(model_class, **init_kwargs)

    if model_weights is not None:
        model.load_state_dict(model_weights)

    return model
