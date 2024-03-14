from typing import List
from torch import Tensor
from torch.distributions.distribution import Distribution
from function_class import FunctionClass

class ModifiedFunctionClass(FunctionClass):

    def __init__(self, inner_function_class: FunctionClass):
        self._in_fc = inner_function_class

        self.x_dist = self._in_fc.x_dist
        self.p_dist = self._in_fc.p_dist

        self.batch_size = self._in_fc.batch_size
        self.sequence_length = self._in_fc.sequence_length
        self.x_dim = self._in_fc.x_dim
        self.x_curriculum_dim = self._in_fc.x_curriculum_dim
        self.y_dim = self._in_fc.y_dim

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
