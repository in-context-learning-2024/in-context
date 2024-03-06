import torch
from models.context_model import ContextModel
from tqdm import trange

class XGBoostModel(ContextModel):
    def __init__(self):
        super(XGBoostModel, self).__init__()
        self.name = "xgboost"
        self.context_length = -1

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def forward(self, xs, ys):
        xs, ys = xs.cpu(), ys.cpu()


        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in trange(ys.shape[1]):
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

