import torch

from torch import Tensor
from typing import Iterable

from .benchmark import Benchmark
from .metric import Metric
from core import (
    FunctionClass,
    ContextModel,
)
from .function_error import FunctionClassError

class RegressionScore(Benchmark):
    # zero_err is of shape (1)
    # optimal_err is of shape (seq_length,)
    def __init__(self, metric: Metric, function_class: FunctionClass, zero_err, optimal_err):
        self.funct_err = FunctionClassError(metric, function_class)
        self.zero_err = zero_err
        self.optimal_err = optimal_err.mean(dim = 1)

    def evaluate(self, models: Iterable[ContextModel]) -> Iterable[Tensor]:
        # generating model errors
        model_err = torch.Tensor(self.funct_err.evaluate(models))

        # computations for reg_score
        norm_model_err = torch.sub(model_err, self.zero_err)
        norm_optimal_err = torch.sub(self.optimal_err, self.zero_err)

        # avg over batch_size
        avg_model_err = torch.mean(norm_model_err, dim=1)

        # divide over seq_length, then mean over seq_length to get score
        return torch.mean(torch.div(avg_model_err,norm_optimal_err), dim=1)