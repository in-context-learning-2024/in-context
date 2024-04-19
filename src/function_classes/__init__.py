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

from .chebyshev_specials import (
    ChebyshevKernelLinearRegression,
    ChebychevSharedRoots,
)

FUNCTION_CLASSES = {
    "linear regression" : LinearRegression,
    "linear classification" : LinearClassification,
    "sparse linear" : SparseLinearRegression,
    "quadradtic regression" : QuadraticRegression,
    "2 layer mlp regression" : MLPRegression,
    "decision tree" : DecisionTreeRegression,
    "chebyshev kernel linear regression" : ChebyshevKernelLinearRegression,
    "chebychev shared roots" : ChebychevSharedRoots,
}

__all__ = [
    "FUNCTION_CLASSES",
]
