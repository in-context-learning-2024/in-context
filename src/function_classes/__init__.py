from .decision_tree import DecisionTreeRegression
from .linear import (
    LinearRegression, 
    LinearClassification, 
    SparseLinearRegression, 
    QuadraticRegression
)
from .mlp import MLPRegression
from .wrappers import (
    ModifiedFunctionClass,
    NoisyRegression,
    ScaledRegression
)

FUNCTION_CLASSES = {
    "linear regression" : LinearRegression,
    "linear classification" : LinearClassification,
    "sparse linear" : SparseLinearRegression,
    "quadradtic regression" : QuadraticRegression,
    "2 layer mlp regression" : MLPRegression,
    "decision tree" : DecisionTreeRegression
}

__all__ = [
    "FUNCTION_CLASSES",
]
