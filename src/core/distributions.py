import torch
from torch.distributions.distribution import Distribution


class randQuadrant(Distribution):
    def __init__(self, base: Distribution, opp = False):
        bs = base.batch_shape()
        # double the first dim of shape, because the first half will be modified
        self.super(randQuadrant, (bs[0]*2, bs[1], bs[2]), base.event_shape())
        self.dist = base
        # opposite is for if we want ex and test to be opposite quadrants
        self.opposite = opp
    
    def sample(self):
        xs = self.dist.sample()
        pattern = torch.randn([xs.shape[0], 1, xs.shape[2]]).sign()

        xs_modded = xs.abs() * pattern
        
        # first half is modified to be one quadrant
        if (self.opposite):
            return torch.cat((xs_modded, -xs_modded), dim=1)
        return torch.cat((xs_modded, xs), dim=1)
