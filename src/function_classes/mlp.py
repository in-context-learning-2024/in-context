import torch

from torch.distributions.distribution import Distribution
from function_classes.function_class import FunctionClass

class MLPRegression(FunctionClass):
    def __init__(self, x_distribution: Distribution, param_distribution_class: type[Distribution], hidden_dimension: int):
        self._hidden_dim = hidden_dimension
        super(MLPRegression, self).__init__(x_distribution, param_distribution_class)

    def __get_parameter_shape(self, x_dim: int, y_dim: int = 1):
        return torch.Size([x_dim + y_dim, self._hidden_dim])
    
    def evaluate(self, x_batch: torch.Tensor) -> torch.Tensor:
    
        raw_params = self._p_dist.sample().to(x_batch.device)
        input_weight_mat  = raw_params[:, :, :self.x_dim]
        output_weight_mat = raw_params[:, :, self.x_dim:].transpose(-1, -2)

        activations = torch.nn.functional.relu(
            torch.bmm(input_weight_mat, x_batch[..., None])
        )

        y_batch = torch.bmm(output_weight_mat, activations).squeeze()
        assert y_batch.shape == (self.batch_size, self.sequence_length, self.y_dim), \
            f"Produced wrong output shape in MLP function class! Expected: {(self.batch_size, self.sequence_length, self.y_dim)}  Got: {tuple(y_batch.shape)}"

        return y_batch
