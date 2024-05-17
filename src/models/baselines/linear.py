import torch
import warnings

from typing import Optional, Any
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning

from core import Baseline


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel(Baseline):
    def __init__(self, driver: Optional[str] = None, **kwargs: Any):
        super(LeastSquaresModel, self).__init__(**kwargs)

        y_dim = kwargs.get('y_dim', 1)
        if y_dim != 1:
            raise ValueError(f"Least Squares only supports y dimension of 1! Got: {y_dim}")

        self._driver = driver
        self.name = f"OLS_driver={driver}"
        self.context_length = -1

    def evaluate(self, xs: torch.Tensor, ys: torch.Tensor):
        DEVICE = xs.device
        xs, ys = xs.cpu(), ys.cpu()
        ys = ys[..., 0] # remove the trivial y_dim=1 dimension
        
        preds = []

        if xs.shape[-2] not in (ys.shape[-1], ys.shape[-1] + 1):
            raise ValueError(
                "Can only inference with x sequences either 1 longer or as long as y sequences!" + \
                f"Got: X sequences of length {xs.shape[-2]} and Y sequences of lengh {ys.shape[-2]}"
            )

        for i in range(xs.shape[-2]):
            if i == 0:
                preds.append(torch.zeros_like(xs[:, :1, 0], device=xs.device))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self._driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, :1])

        return torch.stack(preds, dim=1).to(device=DEVICE)


class AveragingModel(Baseline):
    def __init__(self, **kwargs: Any):
        super(AveragingModel, self).__init__(**kwargs)
        self.name = "averaging"
        self.context_length = -1

    def evaluate(self, xs: torch.Tensor, ys: torch.Tensor):
        preds = []

        for i in range(ys.shape[1]):
            if i == 0:
                preds.append(torch.zeros_like(xs[:, 0, :self.y_dim], device=xs.device))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, :1])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel(Baseline):
    def __init__(self, alpha: float, max_iter: int = 100000, **kwargs: Any):
        super(LassoModel, self).__init__(**kwargs)

        # the l1 regularizer gets multiplied by alpha.
        self._alpha = alpha
        self._max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"
        self.context_length = -1

    def evaluate(self, xs: torch.Tensor, ys: torch.Tensor):
        DEVICE = xs.device
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
                            clf.fit(train_xs.numpy(), train_ys.numpy())
                        except ConvergenceWarning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1).to(device=DEVICE)
