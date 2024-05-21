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
    # base_err is of shape (seq_length,)
    def __init__(self, metric: Metric, function_class: FunctionClass, zero_err, base_err):
        self.funct_err = FunctionClassError(metric, function_class)
        self.zero_err = zero_err
        self.base_err = base_err

    def evaluate(self, models: Iterable[ContextModel], num_batches: int = 1) -> Iterable[Tensor]:
        # generating model errors
        model_err = torch.tensor(self.funct_err.evaluate(models, num_batches=num_batches))
        model_err = model_err.squeeze()

        # avg across batch_size
        avg_model_err = torch.mean(model_err, dim=1)

        norm_model_err = torch.sub(avg_model_err, self.zero_err)
        norm_base_err = torch.sub(self.base_err, self.zero_err)
        
        # divide across seq_length, then mean over seq_length to get score
        scores = torch.div(norm_model_err, norm_base_err)
        scores = torch.mean(scores[:, 1:], dim=1) # we skip the first point because all models are trivially the zero estimator with no context
        return scores
