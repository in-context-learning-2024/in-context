import torch.nn.functional as F

from torch import Tensor


class Metric:
    def evaluate(self, ground_truth: Tensor, predictions: Tensor) -> Tensor:
        raise NotImplementedError(f"Abstract class Metric does not have an evaluation!")
        
class SquaredError(Metric):
    def evaluate(self, ground_truth: Tensor, predictions: Tensor) -> Tensor:
        return (ground_truth - predictions).square()

class MeanSquaredError(Metric):
    def evaluate(self, ground_truth: Tensor, predictions: Tensor) -> Tensor:
        return (ground_truth - predictions).square().mean()

class Accuracy(Metric):
    def evaluate(self, ground_truth: Tensor, predictions: Tensor) -> Tensor:
        return (ground_truth == predictions.sign()).float()

class CrossEntropy(Metric):
    def evaluate(self, ground_truth: Tensor, predictions: Tensor) -> Tensor:
        output = F.sigmoid(predictions)
        target = (ground_truth + 1) / 2
        return F.binary_cross_entropy(output, target)
