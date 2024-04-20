import torch
from torch.distributions.distribution import Distribution


class randQuadrant(Distribution):
    def __init__(self, base: Distribution):
        bs = base.batch_shape()
        # double the first dim of shape, because the first half will be modified
        self.super(randQuadrant, (bs[0]*2, bs[1], bs[2]), base.event_shape())
        self.dist = base
    
    def sample(self):
        xs = self.dist.sample()
        pattern = torch.randn(xs.shape).sign()

        xs_modded = xs.abs() * pattern

        return torch.cat((xs, xs_modded), dim=1)
