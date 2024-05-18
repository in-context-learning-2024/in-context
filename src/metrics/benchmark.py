
from torch import Tensor
from typing import Iterable

from core import (
    FunctionClass,
    ContextModel,
)

class Benchmark:
    def __init__(self):
        raise NotImplementedError("Abstract Class Benchmark cannot be instantiated!")

    def evaluate(self, models: Iterable[ContextModel]) -> Iterable[Tensor]:
        """Produce a benchmark value for an iterable collection of ContextModels"""
        raise NotImplementedError(f"Abstract Class Benchmark cannot evaluate!")


