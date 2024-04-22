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
        self.fn_cls = function_class
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
                for _, (x_batch, y_batch) in zip(range(num_batches), self.fn_cls)
            ])

            # errs is of shape: (#batches, #models, batch_size, sequence_length, *metric_dims)

        errs = torch.transpose(errs, 0, 1)
        errs = torch.flatten(errs, 1, 2)

        return errs

#################################################################################################################################
    def evaluateRobustness_quadrant(self, models: Iterable[ContextModel], num_batches: int = 1)-> Iterable[Tensor]:
        sequence_length=self.fn_cls.sequence_length
        batch_size=self.fn_cls.batch_size
        models = list(models)
        num_models = len(models)

        # note no sequence length in errors, we only want error of last item
        errs=torch.zeros((num_models, num_batches, batch_size))

        # convert distribution to single quandrant
        quad_fn = self.fn_cls
        quad_fn.x_dist = distributions.randQuadrant(self.fn_cls.x_dist)

        # each batch corresponds to a number of quadrant-restricted values
        for i, (x_batch, y_batch) in zip(range(num_batches), quad_fn):
            # x_modded has x-values that have been restricted to a quadrant
            x_modded = x_batch[:batch_size]
            x = x_batch[batch_size:]

            y_modded = y_batch[:batch_size]
            y = y_batch[batch_size:]

            # want to gradually increase number of quadrant-ed values
            # as i increases, we reduce available clean data
            x_comb = torch.cat((x_modded[:, :sequence_length - 1], x[:, sequence_length - 1:sequence_length]), dim=1)
            y_comb = torch.cat((y_modded[:, :sequence_length - 1], y[:, sequence_length - 1:sequence_length]), dim=1)
            
            for j in range(num_models):
                model = models[j]
                with torch.no_grad():
                    errs[j, i] = self.metric.evaluate(
                        y_comb[:,sequence_length - 1], 
                        model.forward(x_comb, y_comb)[:,sequence_length - 1])

            errs=torch.reshape(errs, (num_models, num_batches*batch_size))
        
        # revert modifications for quad
        self.fn_cls.x_dist = self.fn_cls.x_dist.dist
        self.fn_cls.batch_size = batch_size

        return errs
#################################################################################################################################

    def evaluateOrthogonal(self, models: Iterable[ContextModel], num_batches: int = 1)-> Iterable[Tensor]:
        
        sequence_length=self.fn_cls.sequence_length
        batch_size=self.fn_cls.batch_size
        num_models = len(list(models))
        errs=torch.zeros((num_models, num_batches, batch_size, sequence_length))


        for i in range(num_batches):
            params=self.fn_cls.p_dist.sample()
            x_batch=self.fn_cls.x_dist.sample()
            n=x_batch.shape[2]
            
            A = torch.randn(batch_size, n, n)
            Q, _ = torch.linalg.qr(A, mode="complete")
            context_space=Q.clone() #discuss with sahai 
            context_space[:, :, -1]=0
            test_space=Q.clone()
            test_space[:, :, :-1]=0
            x_context, x_test=torch.zeros_like(x_batch), torch.zeros_like(x_batch)
            for j in range(batch_size):

                x_context[j]=x_batch[j] @context_space[j]
                x_test[j]=x_batch[j] @test_space[j]

            for j in range(1, sequence_length):
                
                cur_x=x_context.clone()
                cur_x[:,j]=x_test[:, j]

                if isinstance(params, list):
                    y_test=self.fn_cls.evaluate(cur_x, *params)
                else:
                    y_test=self.fn_cls.evaluate(cur_x, params)
                
                with torch.no_grad():
                    errs[:, i, :, j] = torch.stack([
                  
                        self.metric.evaluate(
                            y_test,
                            model.forward(cur_x, y_test)
                        )
                        for model in models
                    ])[:, :, j]
        
        errs=torch.reshape(errs, (num_models, num_batches*batch_size, sequence_length))[:, :, 1:]

        return errs

    def evaluateAtSeenPoints(self, models: Iterable[ContextModel], num_batches: int = 1)-> Iterable[Tensor]:#each evaluation happens at a random already seen point. 
        
        sequence_length=self.fn_cls.sequence_length
        batch_size=self.fn_cls.batch_size
        num_models = len(list(models))
        errs=torch.zeros((num_models, num_batches, batch_size, sequence_length))

        for i in range(num_batches):
            params=self.fn_cls.p_dist.sample()
            x_batch=self.fn_cls.x_dist.sample()
            
            for j in range(1, sequence_length):
                x_test=x_batch.clone()
                perm = torch.stack([torch.randperm(j) for _ in range(batch_size)]).unsqueeze(dim=1)
                ind_mat = (perm == 0) + 0.0
                x_test[:, j:j+1] = ind_mat @ x_batch[:, :j]

                if isinstance(params, list):
                    y_test=self.fn_cls.evaluate(x_test, *params)
                else:
                    y_test=self.fn_cls.evaluate(x_test, params)
                
                with torch.no_grad():
                    errs[:, i, :, j] = torch.stack([
                  
                        self.metric.evaluate(
                            y_test,
                            model.forward(x_test, y_test)
                        )
                        for model in models
                    ])[:, :, j]
        
        errs=torch.reshape(errs, (num_models, num_batches*batch_size, sequence_length))[:, :, 1:]

        return errs

    def evaluateFLOPS(self, models: Iterable[ContextModel])-> Iterable[Tensor]:
        raise NotImplementedError #interface for other architechture group
        return None
    
    def evaluateAccumulationPerSec(self, models: Iterable[ContextModel])-> Iterable[Tensor]:
        raise NotImplementedError #interface for architechture group
        return None

class FCErrorQuadrants(FunctionClassError):
    """Determine error of models on a function class when the context has a constant
       sign per sequence and the query point either has 
       has each component have a sign opposite to that of all context examples
    """

    def __init__(self, metric: Metric, function_class: FunctionClass, opposite: bool = True):
        super().__init__(metric, function_class)
        self.opposite = opposite

    def evaluate(self, models: Iterable[ContextModel], num_batches: int = 1) -> Iterable[Tensor]:

        batch_size = self.fn_cls.batch_size
        y_dim = self.fn_cls.y_dim
        models = list(models)
        num_models = len(models)

        # note no sequence length in errors, we only want error of last item
        errs = torch.zeros((num_batches, num_models, batch_size, y_dim))

        for batch_num in range(num_batches):
            xs = self.fn_cls.x_dist.sample()
            pattern = torch.randn(xs[:, 0:1, :].shape).sign() # shape (1, sequence_len, x_dim)

            xs_context = xs.abs() * pattern
            assert xs_context.shape == xs.shape

            params: list[Tensor] | Tensor  = self.fn_cls.p_dist.sample()
            x_query = (-xs_context if self.opposite else xs)[:, -1:]
            if isinstance(params, list):
                ys_context = self.fn_cls.evaluate(xs_context, *params)
                y_query = self.fn_cls.evaluate(x_query, *params)
            else:
                ys_context = self.fn_cls.evaluate(xs_context, params)
                y_query = self.fn_cls.evaluate(x_query, params)

            x_comb = torch.cat((xs_context[:, :-1], x_query), dim=1)

            with torch.no_grad():
                errs[batch_num] = torch.stack([
                    self.metric.evaluate(
                        y_query.unsqueeze(dim=-1), # shape (batch_size, y_dim)
                        model.forward(x_comb, ys_context[:, :-1])[:, -1:] # shape (batch_size, y_dim)
                    ) for model in models
                ])

        errs = torch.transpose(errs, 0, 1)
        errs = torch.reshape(errs, (num_models, num_batches*batch_size, y_dim))

        return errs
