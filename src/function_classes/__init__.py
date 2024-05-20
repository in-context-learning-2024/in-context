from core.function_class import FunctionClass
from .decision_tree import DecisionTreeRegression
from .linear import (
    LinearRegression, 
    LinearClassification, 
    SparseLinearRegression, 
    QuadraticRegression,
)
from .mlp import MLPRegression
from .sparse_parity import SparseParityRegression
from .retrieval import Retrieval
from .wrappers import (
    NoisyRegression,
    ScaledRegression,
)

from .chebyshev import (
    ChebyshevKernelLinearRegression,
    ChebyshevSharedRoots,
)

FUNCTION_CLASSES: dict[str, type[FunctionClass]] = {
    "linear regression" : LinearRegression,
    "linear classification" : LinearClassification,
    "sparse linear regression" : SparseLinearRegression,
    "quadratic regression" : QuadraticRegression,
    "2 layer mlp regression" : MLPRegression,
    "decision tree" : DecisionTreeRegression,
    "sparse parity regression" : SparseParityRegression,
    "retrieval" : Retrieval,

    "chebyshev kernel linear regression" : ChebyshevKernelLinearRegression,
    "chebyshev shared roots" : ChebyshevSharedRoots,
}

__all__ = [
    "FUNCTION_CLASSES",
]
