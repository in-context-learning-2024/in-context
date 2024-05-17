import torch
from torch import nn, Tensor
from typing import Optional

from .errors import ShapeError

class ContextModel:
    def __init__(self, x_dim: int, y_dim: int = 1):
        super().__init__()
        self.context_length: Optional[int] = None
        self.name: str = "Unknown_ContextModel"
        self.y_dim = y_dim
        self.x_dim = x_dim

    def __repr__(self):
        return self.name

    def get_dims(self, xs: Tensor, ys: Tensor) -> tuple[int, int, tuple[int, int]]:

        x_shape_err_msg = " \n\t".join([
            "", "Expected: (batch_size, seq_len, x_dim)",
            f"Got: {', '.join(map(str, xs.shape))}"
        ])
        y_shape_err_msg = " \n\t".join([
            "", "Expected: (batch_size, seq_len, x_dim)",
            f"Got: {', '.join(map(str, ys.shape))}"
        ])
        
        if len(xs.shape) < 3:
            raise ShapeError("Not enough x dimensions" + x_shape_err_msg)
        elif len(xs.shape) > 3:
            raise ShapeError("Too many x dimensions!" + x_shape_err_msg)
        
        if len(ys.shape) < 3:
            raise ShapeError("Not enough y dimensions" + y_shape_err_msg)
        elif len(ys.shape) > 3:
            raise ShapeError("Too many y dimensions!" + y_shape_err_msg)

        x_bs, x_seq, x_dim = xs.shape
        y_bs, y_seq, y_dim = ys.shape

        if x_bs != y_bs:
            raise ShapeError(
                f"Batch sizes for x and y don't match! x: {x_bs}, y: {y_bs}"
            )
        if x_seq not in (y_seq, y_seq+1):
            raise ShapeError(" \n\t".join([
                f"Unexpected number of y values!",
                f"Expected: {(x_seq, x_seq-1)}",
                f"Got: {y_seq}"
            ]))

        if x_dim < y_dim:
            raise ShapeError( 
                "y dimension cannot be larger than x dimension! "
                + f"Got: x_dim = {x_dim}, y_dim = {y_dim}" 
            )

        if x_dim != self.x_dim:
            raise ShapeError(
                f"Unexpected x dimension! Expected: {self.x_dim} Got: {x_dim}"
            )

        if y_dim != self.y_dim:
            raise ShapeError(
                f"Unexpected y dimension! Expected: {self.y_dim} Got: {y_dim}"
            )

        return (x_bs, x_seq, (x_dim, y_dim))

    def evaluate(self, xs: Tensor, ys: Tensor) -> Tensor:
        """
        Translate from a sequence of x,y pairs to predicted y 
        values for each presented x value. `xs` must be the 
        same length as or exactly one longer than `ys`
        """
        raise NotImplementedError(f"Abstract class ContextModel does not implement `.evaluate()`!")

    # Helper for .forward
    def interleave(self, xs: Tensor, ys: Tensor) -> Tensor:
        # code adapted from Garg et. al.
        """Interleaves the x's and the y's into a single sequence with shape (batch_size, 2*num_points, x_dim)"""
        bsize, points, (x_dim, y_dim) = self.get_dims(xs, ys)

        ys_wide = torch.cat(
            (
                ys.view(bsize, points, y_dim),
                torch.zeros(bsize, points, x_dim - y_dim, device=ys.device),
            ),
            dim=2,
        )
        zs = torch.stack((xs, ys_wide), dim=2)
        zs = zs.view(bsize, 2 * points, x_dim)
        return zs

    @staticmethod    # Helper for .forward
    def stack(xs: Tensor, ys: Tensor, ctx_len: int = 1) -> Tensor:
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


class Baseline(ContextModel):
    def __init__(self, x_dim: int, y_dim: int = 1):
        super().__init__(x_dim, y_dim=y_dim)

class TrainableModel(ContextModel, nn.Module):
    def __init__(self, x_dim: int, y_dim: int = 1):
        super().__init__(x_dim, y_dim=y_dim)

    def forward(self, xs: Tensor, ys: Tensor) -> Tensor:
        raise NotImplementedError(f"Abstract TrainableModel does not implement .forward()!")
    
    def evaluate(self, xs: Tensor, ys: Tensor) -> Tensor:
        return self(xs, ys)
