import yaml
import torch.distributions as D
import torch

from typing import Any, Callable, Optional

from models import MODELS
from function_classes import FUNCTION_CLASSES
from core import ContextModel, FunctionClass
from utils import curried_throw

class Curriculum(yaml.YAMLObject):

    yaml_tag = u'!curriculum'

    def __init__(
            self,
            step_len : int,
            start : int | float,
            stop : int | float,
            step_size : int | float,
        ):
        self.start = start
        self.stop = stop
        self.step_size = step_size
        self.step_len = step_len

    @property
    def max_phases(self) -> int:
        return int((self.stop - self.start) / self.step_size)

    @property
    def partitioning_steps(self):
        return [ self.step_len * i for i in range(1, self.max_phases + 1) ]

    def __repr__(self):
        return f"Curriculum({self.start} to {self.stop}," + \
                f"step_size={self.step_size}, step_length={self.step_len})"

def _get_value(
    obj: int | float | list | dict | str | Curriculum, 
    step_num: int
):
    if not isinstance(obj, (Curriculum, dict, list)):
        return obj
    elif isinstance(obj, dict):
        return {
            key : _get_value(val, step_num) 
            for key, val in obj.items()
        }
    elif isinstance(obj, list):
        return [ _get_value(item, step_num) for item in obj ]

    phases_past = step_num // obj.step_len
    effective_phases = min(phases_past, obj.max_phases)
    result = obj.start + obj.step_size * effective_phases
    casted_result = type(obj.start)(result)
    return casted_result


def expand_curriculum(raw_data: dict) -> tuple[list[dict], list[int]]:

    def identify_curriculum_params(data: dict) -> list[list]:
        paths = [ ]
        for key, val in data.items():
            if isinstance(val, Curriculum):
                paths.extend([ [key, val] ])
            elif isinstance(val, dict):
                paths.extend([ [ key ]  + path for path in identify_curriculum_params(val) ] )
        return paths

    curriculums = {
        curriculum : route
        for *route, curriculum in identify_curriculum_params(raw_data)
    }

    partitioning_steps = set([ ])
    currs = list(curriculums.keys())
    for c in currs:
        partitioning_steps = partitioning_steps.union(set(c.partitioning_steps))
    partitioning_steps = sorted(list(partitioning_steps))
    partitioning_steps = filter(lambda step_num: step_num <= raw_data['steps'], partitioning_steps)

    stages = [ ]
    durations = [ ]
    last_boundary: int = 0
    for boundary in partitioning_steps:
        dat = _get_value(raw_data, boundary)
        duration = boundary - last_boundary
        last_boundary = boundary
        stages.append(dat)
        durations.append(duration)

    return stages, durations


def get_x_distribution(batch_size: int, seq_len: int, x_dim: int, data: dict) -> Optional[D.Distribution]:
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

    try:
        return dist_class(**kwargs)
    except Exception as e:
        print(f"Unexpected error when instantiating distribution!: \n\t{e}")


def get_model(data: dict) -> Optional[ContextModel]:

    if 'type' not in data:
        raise KeyError(f"Model type not specified!")

    model_class: type[ContextModel] | Callable = MODELS.get(data['type'], 
        curried_throw(NotImplementedError(f"Invalid model type! Got: `{data['type']}`"))
    )


    del data['type']

    try:
        return model_class(**data)
    except Exception as e:
        print(f"Unexpected error when instantiating model!: \n\t{e}")


def get_function_class(x_dist: D.Distribution, x_curr_dim: int, data: dict) -> Optional[FunctionClass]:
    if 'type' not in data:
        raise KeyError(f"Function Class type not specified!")

    f_class_type: type[FunctionClass] | Callable[[], None] = FUNCTION_CLASSES.get(data['type'], 
        curried_throw(NotImplementedError(f"Invalid function class! Got: `{data['type']}`"))
    )

    data.update({
        "x_distribution" : x_dist,
        "x_curriculum_dim" : x_curr_dim,
    })

    del data['type']
    try:
        return f_class_type(**data)
    except Exception as e:
        print(f"Unexpected error when instantiating model!: \n\t{e}")

def get_optimizer(model: ContextModel, data: dict) -> Optional[torch.optim.Optimizer]:
    if 'type' not in data:
        raise KeyError(f"Optimizer type not specified!")

    OPTIMIZERS = {
        "sgd" : torch.optim.SGD,
        "adam": torch.optim.Adam
    }

    optim_type: type[torch.optim.Optimizer] | Callable[[Any], None] = OPTIMIZERS.get(data['type'],
        curried_throw(NotImplementedError(f"Invalid optimizer! Got: `{data['type']}`"))
    )

    del data['type']
    try:
        return optim_type(model.parameters(), **data)
    except Exception as e:
        print(f"Unexpected error when instantiating optimizer!: \n\t{e}")

def get_loss_fn(data: dict) -> Optional[torch.nn.Module]:
    if 'type' not in data:
        raise KeyError(f"Loss function type not specified!")
    
    LOSS_FNS = {
        "squared" : torch.nn.MSELoss
    }

    loss_fn_type: type[torch.nn.Module] | Callable[[], None] = LOSS_FNS.get(data['type'],
        curried_throw(NotImplementedError(f"Invalid option for loss function! Got: {data['type']}"))
    )

    del data['type']
    try:
        return loss_fn_type(**data)
    except Exception as e:
        print(f"Unexpected error when instantiating loss function!: \n\t{e}")


# class ContextTrainer:
#     def __init__(self, *args, **kwargs):
#         print(kwargs)
#         print(f"For step_count: {kwargs['steps']}")

from train.context_trainer import ContextTrainer

def produce_trainer_stages(data: dict) -> tuple[list[ContextTrainer], Optional[ContextModel]]:
    """Convert a list of YAML primitive stage dicts to a list of dictionaries with instantiated objects"""

    x_dim: int = _get_value(data['x_dim'], int(1e99)) 
    stages, step_counts = expand_curriculum(data)
    model = None
    for i in range(len(stages)):
        stages[i]['steps'] = step_counts[i]
    
    for stage in stages:
        b_size  = stage['b_size']
        seq_len = stage['seq_len']
        x_curriculum_dim = stage['x_dim']

        stage['x_dist'] = get_x_distribution(
            b_size, seq_len, x_dim, stage.get('x_dist', {})
        )

        stage['model'] = get_model(
            stage['model'] | { "x_dim" : x_dim }
        )

        model = model or stage['model']

        stage['baseline_models'] = list(map(
            lambda d: get_model(
                d | {"x_dim" : x_dim}
            ), 
            stage['baseline_models']
        ))

        stage['loss_fn'] = get_loss_fn(stage.get('loss_fn', {}))


        if stage['model'] is None:
            raise ValueError("Cannot instantiate optimizer: Model is undefined!")
        stage['optim'] = get_optimizer(stage['model'], stage.get('optim', {}))


        if stage['x_dist'] is None:
            raise ValueError("Cannot instantiate function class: x distribution is undefined!")
        stage['function_class'] = get_function_class(
            stage['x_dist'],
            x_curriculum_dim,
            stage['function_class']
        )

    return [ ContextTrainer(**stage) for stage in stages ], model

def parse_training(filename: str) -> tuple[list[ContextTrainer], str]:
    with open(filename, 'r') as f:
        content = f.read()

    d = yaml.load(content, Loader=yaml.Loader)
    reingestible_yaml = yaml.dump(d, Dumper=yaml.Dumper)

    trainers = produce_trainer_stages(d['train'])

    return trainers, reingestible_yaml

# def train_constructor(loader: yaml.Loader | yaml.FullLoader | yaml.UnsafeLoader, node: yaml.Node) -> Optional[ContextTrainer]:

    # import code
    # code.interact(local=locals())

    # x_dim: int = _get_value(d['train']['x_dim'], int(1e99)) 
    # stages, step_counts = expand_curriculum(d, int(1e4)) # TODO: pull from dict
    # stages = elaborate_stages(stages, x_dim)

    # reingestible_yaml = yaml.dump(d, Dumper=yaml.Dumper)dir(l)

from train.context_trainer import TrainerSteps

# yaml.add_constructor("!trainer", train_constructor)

# stages, yaml_str = parse_training("sample.yml")
# trainer = TrainerSteps(stages)
# trainer.train()

