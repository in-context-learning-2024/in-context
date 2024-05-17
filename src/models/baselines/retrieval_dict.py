import torch
from core import Baseline

from typing import Any

class RetrievalDictModel(Baseline):
    
    def __init__(self, **kwargs: Any):
        self.name = f"retrieval_dict"
        pass

    def evaluate(self, xs: torch.Tensor, ys: torch.Tensor):
        # print("we're going in")
        # print(xs.shape)
        # keys_b, values_b = xs[:, :-1:2, :], xs[:, 1::2, :]
        # print(keys_b.shape)
        # print(values_b.shape)
        # query_b = xs[:, -1, :].unsqueeze(dim=1)
        # ans_list = []
        # print("#keys:", len(keys_b[0]))
        # for i in range(len(xs)):
        #     this_dict = {}
        #     for j in range(len(keys_b[0])):
        #         this_dict[keys_b[i][j]] = values_b[i][j]
        #     print("this_dict:", this_dict)
        #     print("query_b[i]:", query_b[i])
        #     ans_list.append(this_dict[query_b[i]])
        # print("we made it out")
        # return torch.tensor(ans_list)

        xs = xs.to('cpu')

        keys_b, values_b = xs[:, :-1:2, :], xs[:, 1::2, :]
        query_b = xs[:, -1, :].unsqueeze(dim=1)
        inner_products = torch.bmm(query_b, keys_b.transpose(1, 2)).squeeze(1)
        _, retrieval_inds = torch.max(inner_products, dim=1)
        retrieval_inds = retrieval_inds.view(-1, 1, 1).expand(-1, -1, values_b.size(-1))
        y_batch = torch.gather(values_b, 1, retrieval_inds).squeeze(1)
        
        return y_batch.to('cpu')
