import torch
from core import Baseline

from typing import Any

class RetrievalDictModel(Baseline):
    
    def __init__(self, **kwargs: Any):
        super(RetrievalDictModel, self).__init__(**kwargs)
        self.name = f"retrieval_dict"

    def evaluate(self, xs: torch.Tensor, ys: torch.Tensor):
        keys_b, values_b = xs[:, :-1:2, :], xs[:, 1::2, :]
        query_b = xs[:, -1, :].unsqueeze(dim=1)
        inner_products = torch.bmm(query_b, keys_b.transpose(1, 2)).squeeze(1)
        _, retrieval_inds = torch.max(inner_products, dim=1)
        retrieval_inds = retrieval_inds.view(-1, 1, 1).expand(-1, -1, values_b.size(-1))
        y_batch = torch.gather(values_b, 1, retrieval_inds).squeeze(1)
        
        return y_batch
