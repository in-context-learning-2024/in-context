from core import ContextModel
from .transformer import (
    GPT2,
    Llama,
    Mamba,
)

from .hybrid import (
    HybridModel,
)

from .baselines import (
    LeastSquaresModel,
    KNNModel,
    AveragingModel,
    LassoModel,
    GDModel,
    DecisionTreeModel,
    XGBoostModel,
    RetrievalDictModel,
    ZeroModel,
)

MODELS: dict[str, type[ContextModel]] = {
    "gpt2" : GPT2,
    "llama" : Llama,
    "mamba" : Mamba,

    "hybrid" : HybridModel,

    "least squares"  : LeastSquaresModel,
    "knn"            : KNNModel,
    "averaging"      : AveragingModel,
    "lasso"          : LassoModel,
    "grad mlp"       : GDModel,
    "decision tree"  : DecisionTreeModel,
    "xgboost"        : XGBoostModel,
    "retrieval dict" : RetrievalDictModel,
    "zero"           : ZeroModel,
}

__all__ = [
    "MODELS",
]
