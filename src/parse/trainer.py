import yaml
import torch
import torch.distributions as D

from typing import Any, Callable

from train import TrainerSteps
from models import MODELS
from function_classes import FUNCTION_CLASSES
from core import ContextModel, FunctionClass
from utils import curried_throw

from .curriculum import expand_curriculum, get_value

class ParsingError(Exception):
    pass

def _clean_instantiate(class_type, *pass_args, **pass_kwargs):
    
    if 'type' in pass_kwargs:
        del pass_kwargs['type']

    try:
        return class_type(*pass_args, **pass_kwargs)
    except Exception as e:
        raise ParsingError(f"Unexpected error when instantiating {class_type}!: \n\t{e}")

def _check_kwargs(type_mapping: dict, kwarg_dict: dict, display_name: str) -> None:
    if 'type' not in kwarg_dict:
        raise KeyError(f"{display_name} type not specified!")

    if kwarg_dict['type'] not in type_mapping:
        raise NotImplementedError(f"Invalid {display_name}! Got: `{kwarg_dict['type']}`")

def get_x_distribution(batch_size: int, seq_len: int, x_dim: int, data: dict) -> D.Distribution:
    batch_shape = torch.Size([batch_size, seq_len])
    event_shape = torch.Size([x_dim, ])
    full_shape = batch_shape + event_shape


    DISTRIBUTION_BANK: dict[str, tuple[type[D.Distribution], dict[str, Any]]] = {
        "normal" : (
            D.MultivariateNormal,
            { "loc" : torch.zeros(full_shape),
              "covariance_matrix" : torch.eye(x_dim),
            }
        ),
        "uniform" : (
            D.Uniform,
            { "low"  : -torch.ones(full_shape),
              "high" : torch.ones(full_shape),
            }
        ),
    }

    dist_info: tuple[type[D.Distribution], dict[str, Any]] = DISTRIBUTION_BANK[data.get('type',"normal")]
    dist_class, kwargs = dist_info

    del data['type']
    kwargs.update(data)
    kwargs.update({ "validate_args" : True })

    return _clean_instantiate(dist_class, **kwargs)


def get_model(data: dict) -> ContextModel:

    if 'type' not in data:
        raise KeyError(f"Model type not specified!")

    model_class: type[ContextModel] | Callable = MODELS.get(data['type'], 
        curried_throw(NotImplementedError(f"Invalid model type! Got: `{data['type']}`"))
    )

    return _clean_instantiate(model_class, **data)


def get_function_class(x_dist: D.Distribution, x_curr_dim: int, data: dict) -> FunctionClass:
    _check_kwargs(FUNCTION_CLASSES, data, "function class")
    f_class_type: type[FunctionClass] = FUNCTION_CLASSES[data['type']]

    data.update({
        "x_distribution" : x_dist,
        "x_curriculum_dim" : x_curr_dim,
    })

    return _clean_instantiate(f_class_type, **data)


def get_optimizer(model: ContextModel, data: dict) -> torch.optim.Optimizer:

    OPTIMIZERS = {
        "sgd" : torch.optim.SGD,
        "adam": torch.optim.Adam
    }

    _check_kwargs(OPTIMIZERS, data, "optimizer")

    optim_type: type[torch.optim.Optimizer] = OPTIMIZERS[data['type']]

    return _clean_instantiate(optim_type, model.parameters(), **data)


def get_loss_fn(data: dict) -> torch.nn.Module:
    
    LOSS_FNS = {
        "squared" : torch.nn.MSELoss,
        "mse" : torch.nn.MSELoss
    }

    _check_kwargs(LOSS_FNS, data, "loss function")

    loss_fn_type: type[torch.nn.Module] | Callable[[], None] = LOSS_FNS[data['type']]
    
    return _clean_instantiate(loss_fn_type, **data)


def _produce_trainer_stages(data: dict) -> TrainerSteps:
    """Convert a list of YAML primitive stage dicts to a list of dictionaries with instantiated objects"""

    for key in ['b_size','seq_len', 'steps', 'model', 'loss_fn', 'baseline_models', 'optim']:
        if key not in data:
            raise ValueError(f"{key} not provided in training config!")

    x_dim: int = max(
        get_value(data['x_dim'], data['steps']),
        get_value(data['x_dim'], 0),
    )
    stages, step_counts = expand_curriculum(data)

    _x_dist = get_x_distribution(
        stages[0]['b_size'], stages[0]['seq_len'], x_dim, stages[0].get('x_dist', {})
    )
    model = get_model(stages[0]['model'] | { "x_dim" : x_dim })
    loss_fn = get_loss_fn(stages[0].get('loss_fn', {}))
    log_freq = stages[0].get('log_freq', -1)
    optimizer = get_optimizer(model, stages[0].get('optim', {}))
    baseline_models = list(map(
        lambda d: get_model(
            d | {"x_dim" : x_dim}
        ), 
        stages[0]['baseline_models']
    ))
    f_classes = [ ]

    for stage in stages:
        x_curriculum_dim = stage['x_dim']

        f_classes.append(
            get_function_class(
                _x_dist,
                x_curriculum_dim,
                stage['function_class']
            )
        )

    big_trainer = TrainerSteps(
        function_classes=f_classes,
        steps=step_counts,
        model=model,
        optim=optimizer, 
        loss_fn=loss_fn,
        baseline_models=baseline_models,
        log_freq=log_freq,
    )

    return big_trainer

def parse_training(content: str) -> TrainerSteps:
    d = yaml.load(content, Loader=yaml.Loader)

    big_trainer = _produce_trainer_stages(d['train'])

    return big_trainer

def parse_training_from_file(filename: str) -> tuple[TrainerSteps, str]:
    with open(filename, 'r') as f:
        content = f.read()
    return parse_training(content), content

## an example usage of the above function:
# from parse import parse_training
# trainer, yaml_str = parse_training_from_file("sample.yml")
# trainer.train()
