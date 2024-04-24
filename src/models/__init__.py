from core import ContextModel
from .transformer import (
    GPT2,
    Llama
)
from .lora import Lora
from .linear import (
    LeastSquaresModel,
    AveragingModel,
    LassoModel,
)
from .gradient_mlp import GDModel
from .nearest_neighbors import KNNModel
from .xgboost import XGBoostModel
from .decision_tree import DecisionTreeModel
from .zero_model import ZeroModel

MODELS: dict[str, type[ContextModel]] = {
    "gpt2" : GPT2,
    "llama" : Llama,

    "lora" : Lora,

    "least squares" : LeastSquaresModel,
    "knn"           : KNNModel,
    "averaging"     : AveragingModel,
    "lasso"         : LassoModel,
    "grad mlp"      : GDModel,
    "decision tree" : DecisionTreeModel,
    "xgboost"       : XGBoostModel,
    "zero"          : ZeroModel,
}

__all__ = [
    "MODELS"
]
