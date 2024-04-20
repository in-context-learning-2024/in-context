from torch import Tensor
from torch.distributions.distribution import Distribution
from core import FunctionClass, ModifiedFunctionClass

class NoisyRegression(ModifiedFunctionClass):
    def __init__(
            self, 
            noise_distribution: Distribution,
            inner_function_class: FunctionClass,
        ):
        super(NoisyRegression, self).__init__(inner_function_class)
        self._noise_dist = noise_distribution

class NoisyXRegression(NoisyRegression):

    def evaluate(self, x_batch: Tensor, *params: Tensor) -> Tensor:
        return super().evaluate(x_batch + self._noise_dist.sample(), *params)

class NoisyYRegression(NoisyRegression):

    def evaluate(self, x_batch: Tensor, *params: Tensor) -> Tensor:
        return super().evaluate(x_batch, *params) + self._noise_dist.sample()

class ScaledRegression(ModifiedFunctionClass):
    def __init__(
            self,
            scale: int,
            inner_function_class: FunctionClass
        ):
        super(ScaledRegression, self).__init__(inner_function_class)
        self._scale = scale

class ScaledXRegression(ScaledRegression):

    def evaluate(self, x_batch: Tensor, *params: Tensor) -> Tensor:
        return super().evaluate(self._scale * x_batch, *params)

class ScaledYRegression(ScaledRegression):

    def evaluate(self, x_batch: Tensor, *params: Tensor) -> Tensor:
        return self._scale * super().evaluate(x_batch, *params)
