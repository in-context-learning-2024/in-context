from typing import List
from torch import Tensor
from torch.distributions.distribution import Distribution
from core import FunctionClass, ModifiedFunctionClass

class NoisyRegression(ModifiedFunctionClass):
    def __init__(
            self, 
            output_noise_distribution: Distribution,
            inner_function_class: FunctionClass,
        ):
        super(NoisyRegression, self).__init__(inner_function_class)
        self._out_noise_dist = output_noise_distribution

    def evaluate(self, x_batch: Tensor, params: List[Tensor] | Tensor) -> Tensor:
        y_batch = self._in_fc.evaluate(x_batch, params)
        y_batch_noisy = y_batch + self._out_noise_dist.sample()
        return y_batch_noisy

class ScaledRegression(ModifiedFunctionClass):
    def __init__(
            self,
            scale: int,
            inner_function_class: FunctionClass
        ):
        super(ScaledRegression, self).__init__(inner_function_class)
        self._scale = scale
    
    def evaluate(self, x_batch: Tensor, params: List[Tensor] | Tensor) -> Tensor:
        return self._scale * self._in_fc.evaluate(x_batch, params)
