from core import ContextModel
from .transformer import (
    GPT2,
    Llama
)
from .linear import (
    LeastSquaresModel,
    AveragingModel,
    LassoModel,
)
from .gradient_mlp import GDModel
from .nearest_neighbors import KNNModel
from .xgboost import XGBoostModel
from .decision_tree import DecisionTreeModel

MODELS: dict[str, type[ContextModel]] = {
    "gpt2" : GPT2,
    "llama" : Llama,

    "least squares" : LeastSquaresModel,
    "knn"           : KNNModel,
    "averaging"     : AveragingModel,
    "lasso"         : LassoModel,
    "grad mlp"      : GDModel,
    "decision tree" : DecisionTreeModel,
    "xgboost"       : XGBoostModel
}

__all__ = [
    "MODELS"
]
