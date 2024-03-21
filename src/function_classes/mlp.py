import torch
import torch.distributions as D

from core import FunctionClass

class MLPRegression(FunctionClass):
    def __init__(self, hidden_dimension: int, *args, **kwargs):
        self._hidden_dim = hidden_dimension
        super(MLPRegression, self).__init__(*args, **kwargs)

    def _init_param_dist(self) -> D.Distribution:
        """Produce the distribution with which to sample parameters"""
        batch_shape = self.x_dist.batch_shape[:2]
        param_event_shape = torch.Size([self.x_dim + self.y_dim,self._hidden_dim])

        param_dist_shape = torch.Size(batch_shape + param_event_shape)

        param_dist = D.Normal( torch.zeros(param_dist_shape), 
                               torch.ones(param_dist_shape)   )

        return param_dist

    def evaluate(self, x_batch: torch.Tensor, raw_params: torch.Tensor) -> torch.Tensor:
        input_weight_mat  = raw_params[:, :, :self.x_dim]
        output_weight_mat = raw_params[:, :, self.x_dim:].transpose(-1, -2)

        activations = torch.nn.functional.relu(
            torch.bmm(input_weight_mat, x_batch[..., None])
        )

        y_batch = torch.bmm(output_weight_mat, activations).squeeze()
        assert y_batch.shape == (self.batch_size, self.sequence_length, self.y_dim), \
            f"Produced wrong output shape in MLP function class!" + \
            f"Expected: {(self.batch_size, self.sequence_length, self.y_dim)}" + \
            f"Got: {tuple(y_batch.shape)}"

        return y_batch
