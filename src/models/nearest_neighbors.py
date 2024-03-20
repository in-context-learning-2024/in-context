import torch
from core import ContextModel


class KNNModel(ContextModel):
    def __init__(self, n_neighbors, weights="uniform", **kwargs):
        # should we be picking k optimally
        self._n_neighbors = n_neighbors
        self._weights = weights
        self.name = f"KNN_n={n_neighbors}_{weights}"
        self.context_length = -1

    def forward(self, xs, ys):
        
        preds = []

        for i in range(ys.shape[1]):
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
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
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)
