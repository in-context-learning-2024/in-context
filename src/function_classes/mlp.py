import torch

from function_classes.function_class import FunctionClass

class MLPRegression(FunctionClass):
    def __init__(self, hidden_dimension: int, *args, **kwargs):
        self._hidden_dim = hidden_dimension
        super(MLPRegression, self).__init__(*args, **kwargs)

    @property
    def _parameter_shape(self):
        return torch.Size([self.x_dim + self.y_dim, self._hidden_dim])

    def evaluate(self, x_batch: torch.Tensor, raw_params: torch.Tensor) -> torch.Tensor:
        input_weight_mat  = raw_params[:, :, :self.x_dim]
        output_weight_mat = raw_params[:, :, self.x_dim:].transpose(-1, -2)

        activations = torch.nn.functional.relu(
            torch.bmm(input_weight_mat, x_batch[..., None])
        )

        y_batch = torch.bmm(output_weight_mat, activations).squeeze()
        assert y_batch.shape == (self.batch_size, self.sequence_length, self.y_dim), \
            f"Produced wrong output shape in MLP function class! Expected: {(self.batch_size, self.sequence_length, self.y_dim)}  Got: {tuple(y_batch.shape)}"

        return y_batch
