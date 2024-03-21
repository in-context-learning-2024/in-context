
import torch
import torch.distributions as D

from typing import List

from core import FunctionClass
from utils import CombinedDistribution


class DecisionTreeRegression(FunctionClass):

    def __init__(self, depth: int = 4, *args):
        self._depth = depth
        super(DecisionTreeRegression, self).__init__(*args)

    def _init_param_dist(self) -> D.Distribution:
        # Represent the tree using an array. Root node is at index 0, its 2 children at index 1 and 2...
        # Values correspond to the coordinate used at each node of the decision tree.
        
        # Only indices corresponding to non-leaf nodes are relevant
        # return torch.Size([2 ** (self._depth + 1) - 1])
        s = torch.Size([self.batch_size, 2 ** (self._depth + 1) - 1, self.x_dim])
        condition_indices_dist = D.Categorical(
            torch.full(s, 1 / self.x_dim)
        )

        target_values_dist = D.Normal(
            loc=torch.zeros(s[:-1]), scale=torch.ones(s[:-1])
        )

        return CombinedDistribution(
            condition_indices_dist,
            target_values_dist
        )

    def evaluate(self, x_batch: torch.Tensor, params: List[torch.Tensor]) -> torch.Tensor:
        dt_tensor, target_tensor = params
        y_batch = torch.zeros(*x_batch.shape[:2], device=x_batch.device)
        for i in range(self.batch_size):
            xs_bool = x_batch[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            dt = dt_tensor[i]
            target = target_tensor[i]

            cur_nodes = torch.zeros(self.sequence_length, device=x_batch.device).long()
            for _ in range(self._depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            y_batch[i] = target[cur_nodes]

        return y_batch
