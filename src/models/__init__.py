from core import ContextModel
from .transformer import (
    GPT2,
    Llama,
)

from .hybrid import (
    HybridModel,
)

from .all_mod_archs import (
    ModTransformerModel,
    MambaNoAttentionModel,
    MambaFirstGPT2TransformerModel,
    MambaformerModel,
    ModLlamaModel,
    LlamaMambaModel,
    LlamaSingleMambaModel,
)

from .baselines import (
    LeastSquaresModel,
    KNNModel,
    AveragingModel,
    LassoModel,
    GDModel,
    DecisionTreeModel,
    XGBoostModel,
    ZeroModel,
)

MODELS: dict[str, type[ContextModel]] = {
    "gpt2" : GPT2,
    "llama" : Llama,
    "hybrid" : HybridModel,
    "mambafirstgpt2"        : MambaFirstGPT2TransformerModel,
    "mambaonly"             : MambaNoAttentionModel,
    "mambaformer_classic"   : MambaformerModel,
    "mod_transformer"       : ModTransformerModel,
    "llama_mamba"           : LlamaMambaModel,
    "llama_mod"             : ModLlamaModel,
    "llama_standard_hybrid" : LlamaSingleMambaModel,

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
    "MODELS",
]
