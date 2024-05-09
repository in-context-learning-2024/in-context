from core.function_class import FunctionClass
from .decision_tree import DecisionTreeRegression
from .linear import (
    LinearRegression, 
    LinearClassification, 
    SparseLinearRegression, 
    QuadraticRegression,
)
from .mlp import MLPRegression
from .chebyshev import (
    ChebyshevKernelLinearRegression,
    ChebyshevSharedRoots,
)
from .sparse_parity import SparseParityRegression
from .wrappers import (
    NoisyRegression,
    ScaledRegression,
)

FUNCTION_CLASSES: dict[str, type[FunctionClass]] = {
    "linear regression" : LinearRegression,
    "linear classification" : LinearClassification,
    "sparse linear regression" : SparseLinearRegression,
    "quadratic regression" : QuadraticRegression,
    "2 layer mlp regression" : MLPRegression,
    "decision tree" : DecisionTreeRegression,
    "sparse parity regression" : SparseParityRegression,
    
    "chebyshev kernel linear regression" : ChebyshevKernelLinearRegression,
    "chebyshev shared roots" : ChebyshevSharedRoots,
}

__all__ = [
    "FUNCTION_CLASSES",
]
