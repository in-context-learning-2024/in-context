import torch

from sklearn import tree
from typing import Optional, Any

from core import Baseline

class DecisionTreeModel(Baseline):
    def __init__(self, max_depth: Optional[int] = None, **kwargs: Any):
        super(DecisionTreeModel, self).__init__(**kwargs)

        self._max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"
        self.context_length = -1

    def evaluate(self, xs: torch.Tensor, ys: torch.Tensor):
        xs, ys = xs.cpu(), ys.cpu()

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in range(ys.shape[1]):
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self._max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModelSGN(DecisionTreeModel):
    def __init__(self, **kwargs: Any):
        super(DecisionTreeModelSGN, self).__init__(**kwargs)
        self.name = self.name.replace("decision_tree", "decision_treeSGN")

    def evaluate(self, xs: torch.Tensor, ys: torch.Tensor):
        return super().evaluate(torch.sign(xs), ys)
