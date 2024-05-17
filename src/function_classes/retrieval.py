import torch
import torch.distributions as D

from core import FunctionClass

class Retrieval(FunctionClass):

    def _init_param_dist(self) -> D.Distribution:
        return D.Categorical(torch.Tensor([1]))
    
    def evaluate(self, x_batch: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        assert x_batch.shape[0] == self.batch_size and x_batch.shape[2] == self.x_dim
        assert x_batch.shape[1] % 2 == 1

        keys_b, values_b = x_batch[:, :-1:2, :], x_batch[:, 1::2, :]
        query_b = x_batch[:, -1, :].unsqueeze(dim=1)
        inner_products = torch.bmm(query_b, keys_b.transpose(1, 2)).squeeze(1)
        _, retrieval_inds = torch.max(inner_products, dim=1)
        retrieval_inds = retrieval_inds.view(-1, 1, 1).expand(-1, -1, values_b.size(-1))
        y_batch = torch.gather(values_b, 1, retrieval_inds).squeeze(1)
        
        return y_batch
