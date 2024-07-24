import torch
import torch.distributions as D

from torch import Tensor

from core import FunctionClass
from utils import CombinedDistribution

class Retrieval(FunctionClass):

    def _init_param_dist(self) -> D.Distribution:
        # Params are (1) Values (of shape identical to sampled x values)
        #            (2) a selection of an x/y pair by indexing into sampled xs/ys
        return CombinedDistribution(
            self.x_dist,
            D.Categorical(
                torch.ones([self.sequence_length])
                / self.sequence_length
            # expand by both batch_size and seq_len to avoid inconsistency with CombinedDistribution properties
            ).expand(torch.Size([self.batch_size, self.sequence_length])) 
        )

    def __next__(self) -> tuple[Tensor, Tensor]:
        x_batch_tmp: Tensor = self.x_dist.sample()
        x_batch = torch.zeros_like(x_batch_tmp)
        x_batch[..., :self.x_curriculum_dim] = x_batch_tmp[..., :self.x_curriculum_dim]

        params: list[Tensor] = self.p_dist.sample() # pyright: ignore[reportAssignmentType]
        y_batch, query_idxs, *_ = params
        query_idxs = query_idxs[:, 0] # ignore the excess sampled indices

        selected_x = x_batch[torch.arange(x_batch.shape[0]), query_idxs].unsqueeze(1)
        selected_y = y_batch[torch.arange(y_batch.shape[0]), query_idxs].unsqueeze(1)
        x_batch = torch.cat((x_batch, selected_x), dim=1)
        y_batch = torch.cat((y_batch, selected_y), dim=1)

        magnitude = self.x_dim ** 0.5
        x_batch *= magnitude / torch.norm(x_batch, p=2, dim=-1, keepdim=True)
        y_batch *= magnitude / torch.norm(y_batch, p=2, dim=-1, keepdim=True)

        if torch.cuda.is_available():
            return x_batch.cuda(), y_batch.cuda() 
        else:
            return x_batch, y_batch

    def evaluate(self, x_batch: Tensor, *params: Tensor) -> Tensor:
        raise NotImplementedError(
            f"Function Class Retrieval cannot `.evaluate` because it must mutate sampled x-values!"
        )
