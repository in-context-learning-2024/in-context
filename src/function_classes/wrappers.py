from torch import Tensor
from torch.distributions.distribution import Distribution
from core import FunctionClass, ModifiedFunctionClass

class NoisyRegression(ModifiedFunctionClass):
    """Do not directly instantiate this. Instead, instantiate 
       NoisyXRegression or NoisyYRegression"""
    def __init__(
            self, 
            noise_distribution: Distribution,
            inner_function_class: FunctionClass,
        ):
        super(NoisyRegression, self).__init__(inner_function_class)
        self._noise_dist = noise_distribution

class NoisyXRegression(NoisyRegression):

    def modify_x_post_eval(self, x_batch: Tensor) -> Tensor:
        return x_batch + self._noise_dist.sample()

class NoisyYRegression(NoisyRegression):

    def modify_y(self, y_batch: Tensor) -> Tensor:
        return y_batch + self._noise_dist.sample()

class ScaledRegression(ModifiedFunctionClass):
    """Do not directly instantiate this. Instead, instantiate 
       ScaledXRegression or ScaledYRegression"""
    def __init__(
            self,
            scale: float,
            inner_function_class: FunctionClass
        ):
        super(ScaledRegression, self).__init__(inner_function_class)
        self._scale = scale

class ScaledXRegression(ScaledRegression):
    def modify_x(self, x_batch: Tensor) -> Tensor:
        return self._scale * x_batch

class ScaledYRegression(ScaledRegression):

    def modify_y(self, y_batch: Tensor) -> Tensor:
        return self._scale * y_batch
