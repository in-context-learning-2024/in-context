import torch
from torch import nn
from typing import Optional


class ContextModel(nn.Module):
    def __init__(self, **config):
        super(ContextModel, self).__init__()
        self.context_length: Optional[int] = None
        self.name: str = "Unknown_ContextModel"

    def __repr__(self):
        return self.name

    def forward(self, xs: torch.Tensor, ys: torch.Tensor, **kwargs) -> torch.Tensor:
        """Translate from a sequence of x,y pairs to predicted y 
           values for each presented x value. `xs` must be the 
           same length as or exactly one longer than `ys`
        """
        raise NotImplementedError(f"Abstract class ContextModel does not implement `.forward()`!")

    @staticmethod    # Helper for .forward
    def interleave(xs, ys) -> torch.Tensor:
        # code from Garg et. al.
        """Interleaves the x's and the y's into a single sequence with shape (batch_size, 2*num_points, x_dim)"""
        bsize, points, dim = xs.shape
        ys_wide = torch.cat(
            (
                ys.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys.device),
            ),
            dim=2,
        )
        zs = torch.stack((xs, ys_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    # Helper for .forward
    def stack(self, xs: torch.Tensor, ys: torch.Tensor, ctx_len: int = 1) -> torch.Tensor:
        """Stacks the x's and the y's into a single sequence with shape (batch_size, num_points, x_dim + ctx_len * (y_dim + x_dim). Relies on `self.context_len`"""
        bsize, points, dim = xs.shape
        try:
            y_dim = ys.shape[2]
        except IndexError as e:
            y_dim = 1

        xy_seq = torch.cat(
            (xs, 
             ys.view(ys.shape + (1,))), 
            dim=2
        )

        contexted = [
            torch.cat((torch.zeros(bsize, i, dim+y_dim), xy_seq[:, :-i,:]), dim=1)
            for i in range(1, self.context_len + 1)
        ]

        return torch.cat(contexted + [xs], dim=-1) # returns (b_size, seq_len, x_dim + ctx_len * (x_dim + y_dim))
