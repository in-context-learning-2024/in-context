from core.function_class import FunctionClass
from .decision_tree import DecisionTreeRegression
from .linear import (
    LinearRegression, 
    LinearClassification, 
    SparseLinearRegression, 
    QuadraticRegression,
)
from .mlp import MLPRegression
from .wrappers import (
    NoisyRegression,
    ScaledRegression,
)

from .chebyshev import (
    ChebychevKernelLinearRegression,
    ChebychevSharedRoots,
)

FUNCTION_CLASSES: dict[str, type[FunctionClass]] = {
    "linear regression" : LinearRegression,
    "linear classification" : LinearClassification,
    "sparse linear regression" : SparseLinearRegression,
    "quadratic regression" : QuadraticRegression,
    "2 layer mlp regression" : MLPRegression,
    "decision tree" : DecisionTreeRegression,
    "chebychev kernel linear regression" : ChebychevKernelLinearRegression,
    "chebychev shared roots" : ChebychevSharedRoots,
}

__all__ = [
    "FUNCTION_CLASSES",
]
