import torch

from torch import Tensor
from typing import Iterable

from .benchmark import Benchmark
from .metric import Metric
from core import (
    FunctionClass,
    ContextModel,
)

class FunctionClassError(Benchmark):
    def __init__(self, metric: Metric, function_class: FunctionClass):
        self.function_class = function_class
        self.metric = metric

    def evaluate(self, models: Iterable[ContextModel], num_batches: int = 1) -> Iterable[Tensor]:
        """Produce a tensor of shape (batch_size * num_batches, metric_shape) for each model provided"""

        with torch.no_grad():
            errs = torch.stack([
                torch.stack([
                    self.metric.evaluate(
                        y_batch,
                        model.forward(x_batch, y_batch)
                    )
                    for model in models
                ])
                for _, (x_batch, y_batch) in zip(range(num_batches), self.function_class)
            ])

            # errs is of shape: (#batches, #models, batch_size, sequence_length, *metric_dims)

        errs = torch.transpose(errs, 0, 1)
        errs = torch.flatten(errs, 1, 2)

        return errs


class FCErrorQuadrants(FunctionClassError):
    """For prompt (x1, y1,, ..., xn, yn, xq), where xi[k].sign() ==  xj[k].sign() for all i,j = 1, ..., n,
       measure the error for the model's prediction on xq, where:

       if opposite is True --> xq[k].sign() == -1 * xi[k].sign()

       if opposite is False --> xq[k].sign() is random
    """

    def __init__(self, metric: Metric, function_class: FunctionClass, opposite: bool = True):
        super().__init__(metric, function_class)
        self.opposite = opposite

    def evaluate(self, models: Iterable[ContextModel], num_batches: int = 1) -> Iterable[Tensor]:

        batch_size = self.function_class.batch_size
        sequence_length = self.function_class.sequence_length
        y_dim = self.function_class.y_dim
        models = list(models)
        num_models = len(models)

        errs = torch.zeros((num_batches, sequence_length, num_models, batch_size, y_dim))

        for batch_num in range(num_batches):
            xs = self.function_class.x_dist.sample() # shape (batch_size, sequence_length, x_dim)

            # set sign over a full sequence
            pattern = torch.randn(xs[:, 0:1, :].shape).sign() # shape (batch_size, 1, x_dim)
            xs_context = xs.abs() * pattern
            assert xs_context.shape == xs.shape

            x_queries = (-xs_context if self.opposite else xs)

            ys_context: Tensor  # shape (batch_size, seq_len, y_dim)
            y_query: Tensor     # shape (batch_size,       1, y_dim)

            params: list[Tensor] | Tensor  = self.function_class.p_dist.sample()
            for index in range(sequence_length):
                x_query = x_queries[:, index:index+1]

                if isinstance(params, list):
                    ys_context = self.function_class.evaluate(xs_context, *params)
                    y_query = self.function_class.evaluate(x_query, *params)
                else:
                    ys_context = self.function_class.evaluate(xs_context, params)
                    y_query = self.function_class.evaluate(x_query, params)

                assert y_query.shape == torch.Size((batch_size,       1, y_dim))
                y_query = y_query[:, 0, :]

                x_comb = torch.cat((xs_context[:, :index], x_query), dim=1)

                with torch.no_grad():
                    errs[batch_num, index] = torch.stack([
                        self.metric.evaluate(
                            y_query, # shape (batch_size, y_dim)
                            model.forward(x_comb, ys_context[:, :index])[:, -1] # shape (batch_size, y_dim)
                        ) for model in models
                    ])

        errs = torch.transpose(errs, 1, 3) # shape (#batches, batch_size, num_models, seq_len, metric_dim)
        errs = torch.transpose(errs, 0, 2) # shape (num_models, batch_size, #batches, seq_len, metric_dim)
        errs = torch.flatten(errs, 1, 2)

        return errs


class FCErrorOrthogonal(FunctionClassError):

    def evaluate(self, models: Iterable[ContextModel], num_batches: int = 1) -> Iterable[Tensor]:
        sequence_length = self.function_class.sequence_length
        batch_size = self.function_class.batch_size
        y_dim = self.function_class.y_dim
        num_models = len(list(models))

        errs = torch.zeros((num_models, num_batches, batch_size, sequence_length, y_dim))

        for i in range(num_batches):
            params = self.function_class.p_dist.sample()
            x_batch = self.function_class.x_dist.sample()
            n = x_batch.shape[2]
            
            A = torch.randn(batch_size, n, n)
            Q, _ = torch.linalg.qr(A, mode="complete")
            context_space = Q.clone() #discuss with sahai 
            context_space[:, :, -1] = 0
            test_space = Q.clone()
            test_space[:, :, :-1] = 0
            x_context, x_test = torch.zeros_like(x_batch), torch.zeros_like(x_batch)
            for j in range(batch_size):

                x_context[j] = x_batch[j] @ context_space[j]
                x_test[j] = x_batch[j] @ test_space[j]

            for j in range(1, sequence_length):
                
                cur_x = x_context.clone()
                cur_x[:,j] = x_test[:, j]

                if isinstance(params, list):
                    y_test = self.function_class.evaluate(cur_x, *params)
                else:
                    y_test = self.function_class.evaluate(cur_x, params)
                
                with torch.no_grad():
                    errs[:, i, :, j] = torch.stack([
                        self.metric.evaluate(
                            y_test,
                            model.forward(cur_x, y_test)
                        )
                        for model in models
                    ])[:, :, j]
        
        errs = torch.reshape(errs, (num_models, num_batches*batch_size, sequence_length))[:, :, 1:]

        return errs


class FCErrorSeenPoints(FunctionClassError):

    def evaluate(self, models: Iterable[ContextModel], num_batches: int = 1) -> Iterable[Tensor]:
        sequence_length = self.function_class.sequence_length
        batch_size = self.function_class.batch_size
        y_dim = self.function_class.y_dim
        num_models = len(list(models))
        errs = torch.zeros((num_models, num_batches, batch_size, sequence_length, y_dim))

        for i in range(num_batches):
            params = self.function_class.p_dist.sample()
            x_batch = self.function_class.x_dist.sample()
            
            for j in range(1, sequence_length):
                x_test = x_batch.clone()
                perm = torch.stack([torch.randperm(j) for _ in range(batch_size)]).unsqueeze(dim=1)
                ind_mat = (perm == 0) + 0.0
                x_test[:, j:j+1] = ind_mat @ x_batch[:, :j]

                if isinstance(params, list):
                    y_test = self.function_class.evaluate(x_test, *params)
                else:
                    y_test = self.function_class.evaluate(x_test, params)
                
                with torch.no_grad():
                    errs[:, i, :, j] = torch.stack([
                  
                        self.metric.evaluate(
                            y_test,
                            model.forward(x_test, y_test)
                        )
                        for model in models
                    ])[:, :, j]
        
        errs = torch.reshape(errs, (num_models, num_batches*batch_size, sequence_length))[:, :, 1:]

        return errs
