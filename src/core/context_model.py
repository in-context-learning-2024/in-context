import torch
from torch import nn
from typing import Optional


class ContextModel(nn.Module):
    def __init__(self, x_dim: int, y_dim: int = 1):
        super(ContextModel, self).__init__()
        self.context_length: Optional[int] = None
        self.name: str = "Unknown_ContextModel"
        self.y_dim = y_dim
        self.x_dim = x_dim

    def __repr__(self):
        return self.name

    def forward(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """Translate from a sequence of x,y pairs to predicted y 
           values for each presented x value. `xs` must be the 
           same length as or exactly one longer than `ys`
        """
        raise NotImplementedError(f"Abstract class ContextModel does not implement `.forward()`!")

    @staticmethod    # Helper for .forward
    def interleave(xs, ys) -> torch.Tensor:
        # code from Garg et. al.
        """Interleaves the xs and the ys into a single sequence with shape (batch_size, 2*num_points, x_dim)"""
        bsize, y_points, y_dim = ys.shape
        bsize, x_points, x_dim = xs.shape

        if x_points not in (y_points, y_points+1):
            raise ValueError(f"Cannot interleave {x_points} x points and {y_points} y points")
        elif x_points > y_points:
            return torch.cat(
                ( 
                    ContextModel.interleave(xs[:, :-1], ys),
                    xs[:, -1:] 
                ),
                dim=1
            )

        ys_wide = torch.cat(
            (
                ys.view(bsize, y_points, y_dim),
                torch.zeros(bsize, y_points, x_dim - y_dim, device=ys.device),
            ),
            dim=2,
        )

        zs = torch.stack((xs, ys_wide), dim=2)
        zs = zs.view(bsize, 2 * x_points, x_dim)
        return zs

    @staticmethod    # Helper for .forward
    def stack(xs: torch.Tensor, ys: torch.Tensor, ctx_len: int = 1) -> torch.Tensor:
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
            for i in range(1, ctx_len + 1)
        ]

        return torch.cat(contexted + [xs], dim=-1) # returns (b_size, seq_len, x_dim + ctx_len * (x_dim + y_dim))
