import torch

import xgboost as xgb

from typing import Any

from core import Baseline


class XGBoostModel(Baseline):
    def __init__(self, **kwargs: Any):
        super(XGBoostModel, self).__init__(**kwargs)
        self.name = "xgboost"
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

                    clf = xgb.XGBRegressor()

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)

class XGBoostModelSGN(XGBoostModel):
    def __init__(self, **kwargs: Any):
        super(XGBoostModelSGN, self).__init__(**kwargs)
        self.name = self.name.replace("xgboost", "xgboostSGN")

    def evaluate(self, xs: torch.Tensor, ys: torch.Tensor):
        return super().evaluate(torch.sign(xs), ys)
