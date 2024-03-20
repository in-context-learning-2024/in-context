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


def expand_curriculum(raw_data: dict, total_steps: int) -> tuple[list[dict], list[int]]:

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
    partitioning_steps = filter(lambda step_num: step_num <= total_steps, partitioning_steps)

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


# def get_function_class(func_class: str) -> Optional[FunctionClass]:
#     f_class_type: type[ContextModel] | Callable = FUNCTION_CLASSES.get(func_class, 
#         curried_throw(NotImplementedError(f"Invalid function class! Got: `{func_class}`"))
#     )


#     try:
#         return f_class_type(**data)
#     except Exception as e:
#         print(f"Unexpected error when instantiating model!: \n\t>>> {e.__class__}: {e.args[0]}")

# def get_relevant_baselines(task_name):
#     task_to_baselines = {
#         "linear_regression": [
#             (LeastSquaresModel, {}),
#             (KNNModel, {"n_neighbors": 3}),
#             (AveragingModel, {}),
#         ],
#         "linear_classification": [
#             (KNNModel, {"n_neighbors": 3}),
#             (AveragingModel, {}),
#         ],
#         "sparse_linear_regression": [
#             (LeastSquaresModel, {}),
#             (KNNModel, {"n_neighbors": 3}),
#             (AveragingModel, {}),
#         ]
#         + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
#         "relu_2nn_regression": [
#             (LeastSquaresModel, {}),
#             (KNNModel, {"n_neighbors": 3}),
#             (AveragingModel, {}),
#             (
#                 GDModel,
#                 {
#                     "model_class_name": "mlp",
#                     "model_class_args": {
#                         "in_size": 20,
#                         "hidden_size": 100,
#                         "out_size": 1,
#                     },
#                     "opt_alg_name": "adam",
#                     "batch_size": 100,
#                     "lr": 5e-3,
#                     "num_steps": 100,
#                 },
#             ),
#         ],
#         "decision_tree": [
#             (LeastSquaresModel, {}),
#             (KNNModel, {"n_neighbors": 3}),
#             (DecisionTreeModel, {"max_depth": 4}),
#             (DecisionTreeModel, {"max_depth": None}),
#             (XGBoostModel, {}),
#             (AveragingModel, {}),
#         ],
#     }

#     models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
#     return models



# yaml.add_constructor("distribution", lambda loader, node: get_x_distribution(_, _, _, loader.construct_document(node)))


filename = "sample.yml"
with open(filename, 'r') as f:
    content = f.read()

d = yaml.load(content, Loader=yaml.Loader)

stages, step_counts = expand_curriculum(d, int(1e4))

x_dim = max(map(lambda stage: stage['train']['x_dim'], stages))
for stage in stages:
    b_size  = stage['train']['b_size']
    seq_len = stage['train']['seq_len']
    x_curriculum_dim   = stage['train']['x_dim']

    stage['train']['x_dist'] = get_x_distribution(
        b_size, seq_len, x_dim, stage['train'].get('x_dist', {})
    )

    model_dict = stage['train']['model']
    model_dict.update({ "x_dim" : x_dim })
    stage['train']['model'] = get_model(model_dict)

    # stage['train']['function_class'] = get_function_class(stage['train']['funtion_class'])

# fc_dicts, step_counts = expand_curriculum(d['train'], int(1e4))

# import code
# code.interact(local=locals())
