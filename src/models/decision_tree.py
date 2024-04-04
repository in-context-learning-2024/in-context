import torch
from sklearn import tree
from core import ContextModel

class DecisionTreeModel(ContextModel):
    def __init__(self, max_depth=None, **kwargs):
        super(DecisionTreeModel, self).__init__()

        self._max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"
        self.context_length = -1

    def forward(self, xs, ys):
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


