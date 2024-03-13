import torch

from function_classes.function_class import FunctionClass

class LinearRegressionClass(FunctionClass):

    @staticmethod
    def _parameter_shape(x_dim: int, y_dim: int=1):
        return x_dim

    def evaluate(self, x_batch) -> torch.Tensor:
        params = self._p_dist.sample()

        return torch.bmm(x_batch, params).sum(axis=-1, keepdim=True)
