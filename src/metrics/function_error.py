import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Iterable

from metrics.benchmark import Benchmark
from core import FunctionClass
from core import ContextModel
import numpy as np
import scipy.stats as stats

class FunctionClassError(Benchmark):
    def __init__(self, function_class: FunctionClass, num_batches=1, save_path=None):
        self.fn_cls = function_class
        self.num_batches=num_batches
        self.save_path=save_path

    def _metric(self, ground_truth: Tensor, predictions: Tensor) -> Tensor:
        """Compute a metric between a prediction and a "ground truth" """
        raise NotImplementedError("Abstract class FunctionClassError does not implement a metric!")

    def PostProcessingStats(self, errs, model_names, prefix="", B=1000, confidence_level =[0.01,0.05]):
        stats={} 
        if (len(prefix)>0):
            prefix += "_"

        num_models, samples, sequence_length=errs.size()
        
        #Bootstrapping 
        sample_indices = torch.randint(0, samples, (B, samples)) 
        bootstrap_samples = errs[:,sample_indices,:]
        means = bootstrap_samples.mean(dim=2)
        variance_estimate = means.var(dim=1)


        std=torch.std(errs, dim=1)
        mean =torch.mean(errs, dim=1)
        quantiles=torch.quantile(errs, torch.Tensor([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]), dim=1)
        for i, name in enumerate(model_names):
            stats[prefix+"accuracy_"+name]=mean[i]
            stats[prefix+"std_"+name]= std[i]
            stats[prefix+"std_mean_"+name]= std[i]/np.sqrt(samples)
            stats[prefix+"max_"+name]=quantiles[len(quantiles)-1, i]
            stats[prefix+"min_"+name]=quantiles[0, i]
            for j in range(1, len(quantiles)-1):
                stats[prefix+"quantile_"+name+str([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1][j])]=quantiles[j, i]
            for j in range(0, len(confidence_level)):
                #stats[prefix+"confidence_level"+name+str(confidence_level[j])] = [mean[i]-stats.norm.ppf(confidence_level[j]/2)*variance_estimate[i], mean[i]+stats.norm.ppf(1-confidence_level[j]/2)*variance_estimate[i]] 
                continue #something wrong with bootstrap
            
        if self.save_path!=None:
            os.makedirs(self.save_path, exist_ok=True)
            torch.save(stats, os.path.join(self.save_path,self.fn_cls.name))

        return stats

    def evaluate(self, models: Iterable[ContextModel]):# -> Iterable[Tensor]:

        
        sequence_length=self.fn_cls.sequence_length
        batch_size=self.fn_cls.batch_size
        errs=torch.zeros((len(models), self.num_batches, batch_size, sequence_length))

        for i, (x_batch, y_batch) in zip(range(self.num_batches), self.fn_cls):
            with torch.no_grad():
                errs[:, i] = torch.stack([
                  
                    self._metric(
                        y_batch,
                        model.forward(x_batch, y_batch)
                    )
                    for model in models
                ])
        
        errs=torch.reshape(errs, (len(models), self.num_batches*batch_size, sequence_length))

        return self.PostProcessingStats(errs, [model.name for model in models]), errs

    def NoisyXDistribution(self, models: Iterable[ContextModel], noise_x_func):

        sequence_length=self.fn_cls.sequence_length
        batch_size=self.fn_cls.batch_size
        errs=torch.zeros((len(models), self.num_batches, batch_size, sequence_length))

        for i, (x_batch, y_batch) in zip(range(self.num_batches), self.fn_cls):
            with torch.no_grad():
                errs[:, i] = torch.stack([
                  
                    self._metric(
                        y_batch,
                        model.forward(noise_x_func(x_batch), y_batch)
                    )
                    for model in models
                ])
        
        errs=torch.reshape(errs, (len(models), self.num_batches*batch_size, sequence_length))

        return self.PostProcessingStats(errs, [model.name for model in models]), errs

    def NoisyYDistribution(self, models: Iterable[ContextModel], noise_y_func):

        sequence_length=self.fn_cls.sequence_length
        batch_size=self.fn_cls.batch_size
        errs=torch.zeros((len(models), self.num_batches, batch_size, sequence_length))

        for i, (x_batch, y_batch) in zip(range(self.num_batches), self.fn_cls):
            with torch.no_grad():
                y_batch=noise_y_func(y_batch)
                errs[:, i] = torch.stack([
                    self._metric(
                        y_batch,
                        model.forward(x_batch, y_batch)
                    )
                    for model in models
                ])
        
        errs=torch.reshape(errs, (len(models), self.num_batches*batch_size, sequence_length))

        return self.PostProcessingStats(errs, [model.name for model in models]), errs

    def ScaledX(self, models: Iterable[ContextModel], scale):

        sequence_length=self.fn_cls.sequence_length
        batch_size=self.fn_cls.batch_size
        errs=torch.zeros((len(models), self.num_batches, batch_size, sequence_length))

        for i, (x_batch, y_batch) in zip(range(self.num_batches), self.fn_cls):
            with torch.no_grad():
                errs[:, i] = torch.stack([
                    self._metric(
                        y_batch,
                        model.forward(x_batch*scale, y_batch)
                    )
                    for model in models
                ])
        
        errs=torch.reshape(errs, (len(models), self.num_batches*batch_size, sequence_length))

        return self.PostProcessingStats(errs, [model.name for model in models]), errs

    def ScaledY(self, models: Iterable[ContextModel], scale):

        sequence_length=self.fn_cls.sequence_length
        batch_size=self.fn_cls.batch_size
        errs=torch.zeros((len(models), self.num_batches, batch_size, sequence_length))

        for i, (x_batch, y_batch) in zip(range(self.num_batches), self.fn_cls):
            with torch.no_grad():
                y_batch*=scale
                errs[:, i] = torch.stack([
                    self._metric(
                        y_batch,
                        model.forward(x_batch, y_batch)
                    )
                    for model in models
                ])
        
        errs=torch.reshape(errs, (len(models), self.num_batches*batch_size, sequence_length))

        return self.PostProcessingStats(errs, [model.name for model in models]), errs

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

            errs=torch.reshape(errs, (len(models), self.num_batches*batch_size, sequence_length))

            robustness_nums.update(self.PostProcessingStats(errs, [model.name for model in models], save_path, task[0]+str(task[1])))
        
        return robustness_nums


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
