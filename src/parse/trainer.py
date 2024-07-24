import yaml
import torch
import os.path

from pathlib import Path
from typing import Any, Optional, TypeAlias

from train import TrainerSteps
from core import TrainableModel

from .curriculum import expand_curriculum, get_max_value
from .dist import get_x_distribution
from .model import get_model
from .function_class import get_function_class
from .misc import get_optimizer, get_loss_fn
from .utils import YamlMap

ParsedYamlMap: TypeAlias = YamlMap

DEFAULT_TRAINING_OPTS = {
    "y_dim" : 1,
    "x_dist" : { "type": "normal" },
    "baseline_models" : [],
    "log_freq" : -1,
    "checkpoint_freq" : -1,
    "skip_steps" : 0,
}

NEEDED_TRAINING_DATA = [
        'b_size','seq_len', 'steps', 'model', 'loss_fn', 'optim', 'x_dim'
    ] + list(DEFAULT_TRAINING_OPTS.keys())

EXCESS_KEYS = ["y_dim", "x_dist", "function_class", "b_size", "seq_len", "x_dim"]


def _produce_trainer_stages(data: YamlMap) -> TrainerSteps:
    """Convert a YAML primitive stage dicts to a instantiated Trainer object"""

    for key in NEEDED_TRAINING_DATA:
        if key not in data:
            raise ValueError(f"{key} not provided in training config!")

    total_steps: int = data['steps']
    x_dim: int = int(get_max_value(data['x_dim'], total_steps))
    y_dim: int = int(get_max_value(data['y_dim'], total_steps))

    if 'model_weights' in data and 'optim_state' in data:
        data["model"] = get_model(data['model'], x_dim, y_dim, data['model_weights'])
        if not isinstance(data['model'], TrainableModel):
            raise TypeError(f"Model `{data['model'].name}` is not a TrainableModel!")
        data["optim"] = get_optimizer(data['model'], data['optim'], data['optim_state'])
        del data['model_weights']
        del data['optim_state']
    else:
        data["model"] = get_model(data['model'], x_dim, y_dim)
        if not isinstance(data['model'], TrainableModel):
            raise TypeError(f"Model `{data['model'].name}` is not a TrainableModel!")
        data["optim"] = get_optimizer(data['model'], data['optim'])

    data['loss_fn'] = get_loss_fn(data['loss_fn'])

    data['baseline_models'] = list(map(
        lambda kw: get_model(kw, x_dim, y_dim), 
        data['baseline_models']
    ))

    stages, data['steps'] = expand_curriculum(data)

    data['function_classes'] = [
        get_function_class(
            init_kwargs=stage['function_class'],
            x_dist=get_x_distribution(
                stage['b_size'], stage['seq_len'], x_dim, stage['x_dist']
            ),
            x_curr_dim=stage['x_dim'],
            y_dim=stage['y_dim']
        ) 
        for stage in stages
    ]

    for excess_key in EXCESS_KEYS:
        del data[excess_key]

    big_trainer = TrainerSteps(**data)

    return big_trainer

def parse_training(yaml_content: str, skip_steps: int = 0, model_weights: Optional[Any] = None, 
                   optim_state: Optional[Any] = None) -> tuple[TrainerSteps, ParsedYamlMap]:
    d = yaml.load(yaml_content, Loader=yaml.Loader)

    d['train'] = DEFAULT_TRAINING_OPTS | d['train'] # override defaults with specified opts
    training_data = d['train'].copy()

    if skip_steps > 0:
        training_data |= {'skip_steps': skip_steps}
    if model_weights is not None and optim_state is not None:
        training_data |= {'model_weights': model_weights, 'optim_state': optim_state}

    big_trainer = _produce_trainer_stages(training_data)

    return big_trainer, d['train']

def parse_training_from_file(
        filename: str,
        include: Optional[str],
        checkpoint_path: Optional[str] = None
    ) -> tuple[TrainerSteps, ParsedYamlMap]:

    included = ""
    if include is not None:
        for file in Path(include).rglob("*.yml"):
            with open(file, "r") as f:
                included += f.read() + "\n"

    with open(filename, 'r') as f:
        lines = f.readlines()

    lines = [ ' '*4 + line for line in lines ]
    content = f"train:\n" + "\n".join(lines)

    full_yaml = included.strip() + '\n\n' + content

    if checkpoint_path is None:
        return parse_training(full_yaml)

    latest_checkpoint = torch.load(checkpoint_path, 
                                   map_location=("cuda" if torch.cuda.is_available() else "cpu"))
    model_state = latest_checkpoint['model_state_dict']
    optim_state = latest_checkpoint['optimizer_state_dict']
    latest_step = int(os.path.basename(checkpoint_path).split("_")[-1])

    return parse_training(full_yaml, latest_step, model_state, optim_state)
