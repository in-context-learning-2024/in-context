import torch

from typing import Any

from core import Baseline


class KNNModel(Baseline):
    def __init__(self, n_neighbors: int, weights: str = "uniform", **kwargs: Any):
        super(KNNModel, self).__init__(**kwargs)
        self._n_neighbors = n_neighbors
        self._weights = weights
        self.name = f"KNN_n={n_neighbors}_{weights}"
        self.context_length = -1

    def evaluate(self, xs: torch.Tensor, ys: torch.Tensor):
        
        preds = []

        for i in range(ys.shape[1]):
            if i == 0:
                preds.append(torch.zeros(xs.shape[:1] + torch.Size([1, self.y_dim]), device=xs.device))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self._weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, dim=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self._n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred)[:, None, None])

        return torch.cat(preds, dim=1)
