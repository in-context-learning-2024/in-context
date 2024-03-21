import torch
from core import ContextModel
from sklearn.linear_model import Lasso
import warnings

# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel(ContextModel):
    def __init__(self, driver=None, **kwargs):
        super(LeastSquaresModel, self).__init__()

        self._driver = driver
        self.name = f"OLS_driver={driver}"
        self.context_length = -1

    def forward(self, xs, ys):
        xs, ys = xs.cpu(), ys.cpu()
        
        preds = []

        for i in range(ys.shape[1]):
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self._driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


class AveragingModel(ContextModel):
    def __init__(self, **kwargs):
        super(AveragingModel, self).__init__()
        self.name = "averaging"
        self.context_length = -1

    def forward(self, xs, ys):
        preds = []

        for i in range(ys.shape[1]):
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel(ContextModel):
    def __init__(self, alpha, max_iter=100000, **kwargs):
        super(LassoModel, self).__init__()

        # the l1 regularizer gets multiplied by alpha.
        self._alpha = alpha
        self._max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"
        self.context_length = -1

    def forward(self, xs, ys):
        xs, ys = xs.cpu(), ys.cpu()

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in range(ys.shape[1]):
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self._alpha, fit_intercept=False, max_iter=self._max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)

