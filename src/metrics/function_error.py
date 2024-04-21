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
import numpy as np
from scipy.stats import norm

class FunctionClassError(Benchmark):
    def __init__(self, function_class: FunctionClass, num_batches=1):
        self.fn_cls = function_class
        self.num_batches=num_batches

    def _metric(self, ground_truth: Tensor, predictions: Tensor) -> Tensor:
        """Compute a metric between a prediction and a "ground truth" """
        raise NotImplementedError("Abstract class FunctionClassError does not implement a metric!")

    def post_process_errs(self, errs: Iterable[Tensor], prefix="", bootstrap_subsamples: int = 1000, 
                            confidence_level: list[float] = [0.01, 0.05]) -> Iterable[dict[str, Tensor]]:

        if (len(prefix) > 0):
            prefix += "_"

        for err_tensor in errs:
            
            samples, *_ = err_tensor.size()

            # Bootstrapping 
            sample_indices = torch.randint(0, samples, (bootstrap_subsamples, samples)) 
            bootstrap_samples = err_tensor[:,sample_indices,:]
            means = bootstrap_samples.mean(dim=2)
            std_estimate = means.std(dim=1)


            QUANTILES = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
            std = torch.std(err_tensor, dim=0)
            mean = torch.mean(err_tensor, dim=0)
            quantiles = torch.quantile(
                err_tensor, torch.tensor(QUANTILES), dim=0
            )

            confidence_data = { }
            for level in confidence_level:
                upper = mean + norm.ppf(level/2)
                lower = mean + norm.ppf(1 - level/2)
                normalized_std = std/torch.sqrt(torch.tensor(samples))
                confidence_data[f"{prefix}normal_confidence_level{level}"   ] = [upper *  normalized_std, lower *  normalized_std]
                confidence_data[f"{prefix}bootstrap_confidence_level{level}"] = [upper * std_estimate, lower * std_estimate] 

            yield {
                f"{prefix}accuracy" : mean,
                f"{prefix}std" : std,
                f"{prefix}std_mean" : std / np.sqrt(samples),
                f"{prefix}max" : quantiles[len(quantiles)-1],
                f"{prefix}min" : quantiles[0],
                **confidence_data,
                **{
                    f"{prefix}quantile_{q_interval}" : q_value
                    for q_interval, q_value in zip(QUANTILES[1:-1], quantiles[1:-1])
                },
            }

    def evaluate(self, models: Iterable[ContextModel]) -> Iterable[Tensor]:

        with torch.no_grad():
            errs = torch.stack([
                torch.stack([
                    self._metric(
                        y_batch,
                        model.forward(x_batch, y_batch)
                    )
                    for model in models
                ])
                for _, (x_batch, y_batch) in zip(range(self.num_batches), self.fn_cls)
            ])

            # errs is of shape: (#batches, #models, batch_size, sequence_length, *metric_dims)

        errs = torch.transpose(errs, 0, 1)
        errs = torch.flatten(errs, 1, 2)

        return errs

    def evaluateRobustness(self, models: Iterable[ContextModel], noise_x_func, noise_y_func): #probably should be fased out
        
        robustness_tasks=[]

        robustness_nums={}

        for scale in [0.125, 0.25, 0.5, 2, 4, 8]:
            robustness_tasks.append(["scaled_x", scale])
            robustness_tasks.append(["scaled_y", scale])
        
        for noise in [0.0625, 0.125, 0.25, 0.5, 1]:
            robustness_tasks.append(["noise_x", noise])
            robustness_tasks.append(["noise_y", noise])    


        sequence_length=self.fn_cls.sequence_length
        batch_size=self.fn_cls.batch_size

        for j, task in enumerate(robustness_tasks):
            errs=torch.zeros((samples, self.num_batches, batch_size, sequence_length))
    
            for i, (x_batch, y_batch) in zip(range(self.num_batches), self.fn_cls):

                curxs=x_batch
                curys=y_batch
            
                if task[0]=="scaled_x":
                    curxs*=task[1]
                elif task[0]=="scaled_y":
                    curys*=task[1]
                elif task[0]=="noise_x":
                    curxs=noise_x_func(task[1])(curxs)
                elif task[0]=="noise_y":
                    curys=noise_y_func(task[1])(curys)
                
                with torch.no_grad():
                    errs[:, i] = torch.stack([
                  
                        self._metric(
                            curys,
                            model.forward(curxs, curys)
                        )
                        for model in models
                    ])

            errs=torch.reshape(errs, (len(list(models)), self.num_batches*batch_size, sequence_length))

            robustness_nums.update(self.PostProcessingStats(errs, [model.name for model in models], task[0]+str(task[1])))
        
        return robustness_nums
    
    def evaluateRobustness_quadrant(self, models: Iterable[ContextModel])-> Iterable[Tensor]:
        sequence_length=self.fn_cls.sequence_length
        batch_size=self.fn_cls.batch_size
        num_models = len(list(models))

        # note no sequence length in errors, we only want error of last item
        errs=torch.zeros((num_models, self.num_batches, batch_size))

        # convert distribution to single quandrant
        quad_fn = self.fn_cls
        quad_fn.x_dist = distributions.randQuadrant(self.fn_cls.x_dist)

        # each batch corresponds to a number of quadrant-restricted values
        for i, (x_batch, y_batch) in zip(range(self.num_batches), quad_fn):
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

            errs=torch.reshape(errs, (num_models, self.num_batches*batch_size))
        
        # revert modifications for quad
        self.fn_cls.x_dist = self.fn_cls.x_dist.dist
        self.fn_cls.batch_size = batch_size

        return errs

    def evaluateOrthogonal(self, models: Iterable[ContextModel])-> Iterable[Tensor]:
        
        sequence_length=self.fn_cls.sequence_length
        batch_size=self.fn_cls.batch_size
        num_models = len(list(models))
        errs=torch.zeros((num_models, self.num_batches, batch_size, sequence_length))


        for i in range(self.num_batches):
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
        
        errs=torch.reshape(errs, (num_models, self.num_batches*batch_size, sequence_length))[:, :, 1:]

        return errs

    def evaluateAtSeenPoints(self, models: Iterable[ContextModel])-> Iterable[Tensor]:#each evaluation happens at a random already seen point. 
        
        sequence_length=self.fn_cls.sequence_length
        batch_size=self.fn_cls.batch_size
        num_models = len(list(models))
        errs=torch.zeros((num_models, self.num_batches, batch_size, sequence_length))

        for i in range(self.num_batches):
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
        
        errs=torch.reshape(errs, (num_models, self.num_batches*batch_size, sequence_length))[:, :, 1:]

        return errs

    def evaluateFLOPS(self, models: Iterable[ContextModel])-> Iterable[Tensor]:
        raise NotImplementedError #interface for other architechture group
        return None
    
    def evaluateAccumulationPerSec(self, models: Iterable[ContextModel])-> Iterable[Tensor]
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
