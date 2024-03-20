import torch
from torch import nn
from typing import Literal

from tqdm import tqdm

from core import ContextModel
from utils import curried_throw

def get_activation(act: str) -> nn.Module:
    return {
        "relu" : nn.ReLU,
        "gelu" : nn.GELU,
    }.get(
        act,
        curried_throw(ValueError(f"Activation function {act} not implemented!"))
    )()

class MLP(nn.Module):
    def __init__(self, activation: Literal['relu', 'gelu'] = "relu", dimensions: list = [2,2,2]):
        super(MLP, self).__init__()

        layers = [ ]
        last_dim = dimensions[0]
        for dim in dimensions[1:]:
            layers.append(
                nn.Linear(last_dim, dim)
            )

            last_dim = dim
            layers.append(
                get_activation(activation)
            )
        layers = layers[:-1] # remove the extra activation

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ParallelNetworks(nn.Module):
    def __init__(self, num_models, model_class, init_args):
        super(ParallelNetworks, self).__init__()
        self.nets = nn.ModuleList(
            [ model_class(**init_args) for _ in range(num_models) ]
        )

    def forward(self, xs):
        assert xs.shape[0] == len(self.nets)

        for i in range(len(self.nets)):
            out = self.nets[i](xs[i])
            if i == 0:
                outs = torch.zeros(
                    [len(self.nets)] + list(out.shape), device=out.device
                )
            outs[i] = out
        return outs

# Gradient Descent and variants.
# Example usage: gd_model = GDModel("mlp", {'dimensions': [20, 256, 1]}, opt_alg_name = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel(ContextModel):
    def __init__(
        self,
        model_class_name: Literal["mlp"],
        model_class_args: dict,
        opt_alg_name: Literal["sgd", "adam"]="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
        **kwargs
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd

        model_class = {
            "mlp" : MLP,
            # "parallel" : ParallelNetworks 
        }.get(
            model_class_name, 
            curried_throw(ValueError(f"GDModel does not support \"{model_class_name}\" model!"))
        )

        self._get_new_model = lambda: ParallelNetworks(batch_size, model_class, model_class_args)

        self._opt = lambda params: {
            "sgd" : torch.optim.SGD,
            "adam": torch.optim.Adam  
        }.get(
            opt_alg_name,
            curried_throw(ValueError(f"GDModel does not support \"{opt_alg_name}\" optimizer!"))
        )(params, lr=lr)

        self._loss_fn = {
            "squared" : nn.MSELoss
        }.get(loss_name,
            curried_throw(ValueError(f"GDModel does not support \"{loss_name}\" loss function!"))
        )()
        
        self._batch_size = batch_size
        self._num_steps = num_steps

        self.name = f"gdmodel_model={model_class_name}_model_kwargs={model_class_args}_opt={opt_alg_name}_lr={lr}_bsize={batch_size}_nsteps={num_steps}_loss={loss_name}"
        self.context_length = -1

    def forward(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        ys = ys.to(xs.device)

        assert xs.shape[0] == ys.shape[0] == self._batch_size, \
            f"Input values are not of the right batch size! Expected: `{self._batch_size}' Got: {xs.shape[0]}, {ys.shape[0]}"

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = self._get_new_model()
            optim = self._opt(model.parameters())
            # model.to(DEVICE)
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                # Training loop
                for j in range(self._num_steps):

                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self._batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = self._loss_fn(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()

                            outputs_test = model(test_xs)
                            test_loss = self._loss_fn(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optim.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = self._loss_fn(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optim.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)
