
from torch import Tensor
from typing import Iterable
import torch

from .benchmark import Benchmark
from .metric import Metric
from .function_error import FunctionClassError
from core import (
    FunctionClass,
    ContextModel,
)
from models.zero_model import ZeroModel

class Benchmark:
    def __init__(self):
        raise NotImplementedError("Abstract Class Benchmark cannot be instantiated!")

    def evaluate(self, models: Iterable[ContextModel]) -> Iterable[Tensor]:
        """Produce a benchmark value for an iterable collection of ContextModels"""
        raise NotImplementedError(f"Abstract Class Benchmark cannot evaluate!")

class Regression_score(Benchmark):
    def __init__(self, metric: Metric, function_class: FunctionClass, optimal: ContextModel):
        self.function_class = function_class
        self.metric = metric
        self.optimal = optimal

    def evaluate(self, models: Iterable[ContextModel]) -> Iterable[Tensor]:
        model_funct = FunctionClassError(self.metric, self.function_class)
        model_err = model_funct.evaluate(models)

        zero_err = model_funct.evaluate(ZeroModel())

        optimal_err = model_funct.evaluate(self.optimal)

        return reg_score(model_err, zero_err, optimal_err)
    
# errors for each model, in shape (#models, batch size, seq_len)
# errors for the zero estimator (seq_len,)
# error for the optimal estimator (other batch size, seq_len)
def reg_score(model_err, zero_err, optimal_err):
    norm_model_err = torch.sub(model_err, zero_err)
    norm_optimal_err = torch.sub(optimal_err, zero_err)

    avg_model_err = torch.mean(norm_model_err, dim=(1,2))
    avg_optimal_err = torch.mean(norm_optimal_err)

    return avg_model_err - avg_optimal_err
