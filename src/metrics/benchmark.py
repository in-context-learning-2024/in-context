
from torch import Tensor
from typing import Iterable

from core import ContextModel

class Benchmark:
    def __init__(self):
        if type(self) is Benchmark:
            raise NotImplementedError("Abstract Class Benchmark cannot be instantiated!")
        super().__init__()

    def evaluate(self, models: Iterable[ContextModel]) -> Iterable[Tensor]:
        """Produce a benchmark value for an iterable collection of ContextModels"""
        raise NotImplementedError(f"Abstract Class Benchmark cannot evaluate!")
