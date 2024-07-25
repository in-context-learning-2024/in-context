import torch
from torch import nn
from transformers.activations import ACT2FN
from typing import Literal, Any

from tqdm import tqdm

from core import Baseline

class MLP(nn.Module):
    def __init__(self, activation: Literal['relu', 'gelu'] = "relu", dimensions: list[int] = [2,2,2]):
        super(MLP, self).__init__()

        layers = [ ]
        last_dim = dimensions[0]
        for dim in dimensions[1:]:
            layers.append(
                nn.Linear(last_dim, dim)
            )

            last_dim = dim
            layers.append(ACT2FN[activation])
        layers = layers[:-1] # remove the extra activation

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)

class ParallelNetworks(nn.Module):
    def __init__(self, num_models: int, model_class: type[nn.Module], init_args: dict[str, Any]):
        super(ParallelNetworks, self).__init__()
        self.nets = nn.ModuleList(
            [ model_class(**init_args) for _ in range(num_models) ]
        )

    def forward(self, xs: torch.Tensor):
        assert xs.shape[0] == len(self.nets)

        self.nets.to(xs.device)

        out = self.nets[0](xs[0])
        outs = torch.zeros(
            [len(self.nets)] + list(out.shape), device=out.device
        )

        for i in range(len(self.nets)):
            if i == 0: continue
            out = self.nets[i](xs[i])
            outs[i] = out
        return outs

# Gradient Descent and variants.
# Example usage: gd_model = GDModel("mlp", {'dimensions': [256]}, opt_alg_name = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel(Baseline):
    def __init__(
        self,
        model_class_name: Literal["mlp"],
        model_class_args: dict[str, Any],
        opt_alg_name: Literal["sgd", "adam"]="sgd",
        batch_size: int = 1,
        num_steps: int = 1000,
        lr: float = 1e-3,
        loss_name: Literal["squared"] = "squared",
        **kwargs: Any
    ):
        super(GDModel, self).__init__(**kwargs)

        MODEL_CLASSES = {
            "mlp" : MLP,
        }

        OPTIMS = {
            "sgd" : torch.optim.SGD,
            "adam": torch.optim.Adam  
        }

        LOSS_FNS = {
            "squared" : nn.MSELoss
        }

        if model_class_name not in MODEL_CLASSES:
            raise ValueError(f"GDModel does not support \"{model_class_name}\" model!")
        model_class = MODEL_CLASSES[model_class_name]

        model_class_args = model_class_args | { 
            "dimensions" : [
                self.x_dim, *model_class_args.get("dimensions", []), self.y_dim 
            ]
        }
        self._get_new_model = lambda batch_size: ParallelNetworks(batch_size, model_class, model_class_args)

        if opt_alg_name not in OPTIMS:
            raise ValueError(f"GDModel does not support \"{opt_alg_name}\" optimizer!")
        self._opt = lambda params: OPTIMS[opt_alg_name](params, lr=lr)

        if loss_name not in LOSS_FNS:
            raise ValueError(f"GDModel does not support \"{loss_name}\" loss function!")
        self._loss_fn = LOSS_FNS[loss_name]()
        
        self._nets_maximum_batch_size = batch_size
        self._num_steps = num_steps

        self.name = f"gdmodel_model={model_class_name}_model_kwargs={model_class_args}_opt={opt_alg_name}_lr={lr}_bsize={batch_size}_nsteps={num_steps}_loss={loss_name}"
        self.context_length = -1

    def evaluate(self, xs: torch.Tensor, ys: torch.Tensor, verbose: bool = False, print_step: int = 100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        DEVICE = xs.device
        xs_bsize, seq_len, x_dim = xs.shape
        ys = ys.to(DEVICE)

        inds = range(ys.shape[1])

        preds = []  # predict one for first point

        # i: loop over sequence length
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0, 0])
            model = self._get_new_model(xs_bsize)
            optim = self._opt(model.parameters())
            model.to(DEVICE)
            model.train()
            if i > 0:
                pred = torch.zeros_like(ys[:, 0], device=DEVICE)

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                # Training loop
                for j in range(self._num_steps):

                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self._nets_maximum_batch_size]] = True
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
                    outputs = model(train_xs_cur.detach())
                    loss = self._loss_fn(outputs, train_ys_cur.detach())
                    loss.backward()
                    optim.step()

                model.eval()
                pred = model(test_xs).detach()
 
                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1).unsqueeze(-1)
