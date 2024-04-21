import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Iterable

from metrics.benchmark import Benchmark
from core import (
    FunctionClass,
    ContextModel,
    distributions,
)

class FunctionClassError(Benchmark):
    def __init__(self, function_class: FunctionClass):
        self.fn_cls = function_class

    def _metric(self, ground_truth: Tensor, predictions: Tensor) -> Tensor:
        """Compute a metric between a prediction and a "ground truth" """
        raise NotImplementedError("Abstract class FunctionClassError does not implement a metric!")

    def evaluate(self, models: Iterable[ContextModel], num_batches: int = 1) -> Iterable[Tensor]:
        """Produce a tensor of shape (batch_size * num_batches, metric_shape) for each model provided"""

        with torch.no_grad():
            errs = torch.stack([
                torch.stack([
                    self._metric(
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

    def evaluateRobustness_quadrant(self, models: Iterable[ContextModel], num_batches: int = 1)-> Iterable[Tensor]:
        sequence_length=self.fn_cls.sequence_length
        batch_size=self.fn_cls.batch_size
        num_models = len(list(models))

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
            
            for j in range(len(models)):
                model = models[j]
                with torch.no_grad():
                    errs[j, i] = self._metric(
                        y_comb[:,sequence_length - 1], 
                        model.forward(x_comb, y_comb)[:,sequence_length - 1])

            errs=torch.reshape(errs, (num_models, num_batches*batch_size))
        
        # revert modifications for quad
        self.fn_cls.x_dist = self.fn_cls.x_dist.dist
        self.fn_cls.batch_size = batch_size

        return errs

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
                  
                        self._metric(
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
                  
                        self._metric(
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

class SquaredError(FunctionClassError):

    def _metric(self, ground_truth: Tensor, predictions: Tensor) -> Tensor:
        return (ground_truth - predictions).square()

class MeanSquaredError(FunctionClassError): #i dont really think we need this, I really mostly want the local ones...

    def _metric(self, ground_truth: Tensor, predictions: Tensor) -> Tensor:
        return (ground_truth - predictions).square().mean()

class Accuracy(FunctionClassError):

    def _metric(self, ground_truth: Tensor, predictions: Tensor) -> Tensor:
        return (ground_truth == predictions.sign()).float()

class CrossEntropy(FunctionClassError):

    def _metric(self, ground_truth: Tensor, predictions: Tensor) -> Tensor:
        output = F.sigmoid(predictions)
        target = (ground_truth + 1) / 2
        return F.binary_cross_entropy(output, target)
