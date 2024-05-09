import yaml
import torch
import torch.distributions as D

from typing import Any, Callable, Optional

from train import TrainerSteps
from models import MODELS
from function_classes import FUNCTION_CLASSES
from core import ContextModel, FunctionClass
from utils import SparseDistribution

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
        "sparse": (
            SparseDistribution,
            { "batch_shape" : batch_shape,
              "event_shape" : event_shape,
            }
        ),
    }

    dist_info: tuple[type[D.Distribution], dict[str, Any]] = DISTRIBUTION_BANK[data.get('type',"normal")]
    dist_class, kwargs = dist_info

    del data['type']
    kwargs.update(data)
    kwargs.update({ "validate_args" : True })

    return _clean_instantiate(dist_class, **kwargs)


def get_model(data: dict, x_dim: int, y_dim: int) -> ContextModel:

    _check_kwargs(MODELS, data, "model")

    model_class: type[ContextModel] = MODELS[data['type']]

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
    """Convert a YAML primitive stage dicts to a instantiated Trainer object"""

    for key in ['b_size','seq_len', 'steps', 'model', 'loss_fn', 'optim']:
        if key not in data:
            raise ValueError(f"{key} not provided in training config!")

    x_dim_data: int | dict = data['x_dim']
    x_dim: int = max(
        get_value(x_dim_data, data['steps']), # pyright: ignore[reportArgumentType]
        get_value(x_dim_data, 0) # pyright: ignore[reportArgumentType]
    )

    y_dim_data: int | dict = data.get('y_dim', 1)
    y_dim: int = max(
        get_value(y_dim_data, data['steps']), # pyright: ignore[reportArgumentType]
        get_value(y_dim_data, 0) # pyright: ignore[reportArgumentType]
    )
    stages, step_counts = expand_curriculum(data)

    f_classes = [
        get_function_class(
            get_x_distribution(
                stage['b_size'], stage['seq_len'], x_dim, stage.get('x_dist', {})
            ),
            stage['x_dim'],
            stage['function_class']
        ) 
        for stage in stages
    ]

    model = get_model(stages[0]['model'] | { "x_dim" : x_dim }, x_dim, y_dim)
    optimizer = get_optimizer(model, stages[0]['optim'])
    if 'model_weights' in data and 'optim_state' in data:
        model.load_state_dict(data['model_weights'])
        optimizer.load_state_dict(data['optim_state'])

    loss_fn = get_loss_fn(stages[0]['loss_fn'])
    baseline_models = list(map(
        lambda d: get_model(
            d | {"x_dim" : x_dim},
            x_dim, y_dim
        ), 
        stages[0].get('baseline_models', [])
    ))

    log_freq = stages[0].get('log_freq', -1)
    checkpoint_freq = stages[0].get('checkpoint_freq', -1)
    
    skip_steps = data.get('skip_steps', 0)


    big_trainer = TrainerSteps(
        function_classes=f_classes,
        model=model,
        optim=optimizer, 
        loss_fn=loss_fn,
        steps=step_counts,
        baseline_models=baseline_models,
        log_freq=log_freq,
        checkpoint_freq=checkpoint_freq,
        skip_steps=skip_steps,
    )

    return big_trainer

def parse_training(content: str, skip_steps: int = 0, model_weights: Optional[Any] = None, optim_state: Optional[Any] = None) -> TrainerSteps:
    d = yaml.load(content, Loader=yaml.Loader)

    if skip_steps > 0:
        d['train'] |= {'skip_steps': skip_steps}
    if model_weights is not None and optim_state is not None:
        d['train'] |= {'model_weights': model_weights, 'optim_state': optim_state}

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
