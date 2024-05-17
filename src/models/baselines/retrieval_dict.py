import torch
from core import Baseline

from typing import Any

class RetrievalDictModel(Baseline):
    
    def __init__(self, **kwargs: Any):
        pass

    def evaluate(self, xs: torch.Tensor, ys: torch.Tensor):
        keys_b, values_b = xs[:, -1::2, :], xs[:, 1::2, :]
        query_b = xs[:, -1, :].unsqueeze(dim=1)
        ans_list = []
        for i in range(len(xs)):
            this_dict = {}
            for j in range(len(keys_b[0])):
                this_dict[keys_b[i][j]] = values_b[i][j]
            ans_list.append(this_dict[query_b[i]])
        return torch.tensor(ans_list)