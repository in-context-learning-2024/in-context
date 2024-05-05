from core import Baseline

class ZeroModel(Baseline):
    def __init__(self, **kwargs):
        super(ZeroModel, self).__init__(**kwargs)
        self.name = "zero_model"
        self.context_length = -1

    def evaluate(self, xs, ys):
        return 0 * xs[..., 0:self.y_dim]
