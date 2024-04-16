import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Iterable

from metrics.benchmark import Benchmark
from core import FunctionClass
from src.core import ContextModel

class FunctionClassError(Benchmark):
    def __init__(self, function_class: FunctionClass):
        self.fn_cls = function_class
        super().__init__()

    def _metric(self, ground_truth: Tensor, predictions: Tensor) -> Tensor:
        """Compute a metric between a prediction and a "ground truth" """
        raise NotImplementedError("Abstract class FunctionClassError does not implement a metric!")

    def evaluate(self, models: Iterable[ContextModel]) -> Iterable[Tensor]:

        fc_iterable = iter(self.fn_cls)
        x_batch, y_batch = next(fc_iterable)

        with torch.no_grad():
            errs = [
                self._metric(
                    y_batch,
                    model.forward(x_batch, y_batch)
                )
                for model in models
            ]

        return errs


class SquaredError(FunctionClassError):

    def _metric(self, ground_truth: Tensor, predictions: Tensor) -> Tensor:
        return (ground_truth - predictions).square()

class MeanSquaredError(FunctionClassError):

    def _metric(self, ground_truth: Tensor, predictions: Tensor) -> Tensor:
        return (ground_truth - predictions).square().mean()

class Accuracy(FunctionClassError):

    def _metric(self, ground_truth: Tensor, predictions: Tensor) -> Tensor:
        return (ground_truth == predictions.sign()).float()

class CrossEntropy(FunctionClassError):

    def _metric(self, ground_truth: Tensor, predictions: Tensor) -> Tensor:
        output = F.sigmoid(predictions)
        target = (ground_truth + 1) / 2
        return F.binary_cross_entropy(output, target)
