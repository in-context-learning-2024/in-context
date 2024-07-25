import torch.distributions as D

from typing import Any

from core import FunctionClass
from function_classes import FUNCTION_CLASSES

from .utils import check_kwargs, clean_instantiate

def get_function_class(init_kwargs: dict[str, Any], x_dist: D.Distribution, x_curr_dim: int, y_dim: int = 1) -> FunctionClass:
    check_kwargs(FUNCTION_CLASSES, init_kwargs, "function class")
    f_class_type: type[FunctionClass] = FUNCTION_CLASSES[init_kwargs['type']]

    init_kwargs.update({
        "x_distribution" : x_dist,
        "x_curriculum_dim" : x_curr_dim,
        "y_dim" : y_dim,
    })

    return clean_instantiate(f_class_type, **init_kwargs)
