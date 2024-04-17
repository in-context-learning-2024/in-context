import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Iterable

from metrics.benchmark import Benchmark
from core import FunctionClass
from src.core import ContextModel

class FunctionClassError(Benchmark):
    def __init__(self, function_class: FunctionClass):
        self.fn_cls = function_class
        super().__init__()

    def _metric(self, ground_truth: Tensor, predictions: Tensor) -> Tensor:
        """Compute a metric between a prediction and a "ground truth" """
        raise NotImplementedError("Abstract class FunctionClassError does not implement a metric!")

    def PostProcessingStats(self, errs, model_names, save_path=None, prefix=None):
        stats={}

        num_models, samples, batch_size, sequence_length=errs.size()
        
        if (prefix!=None):
            prefix+="_"

        std=torch.std(errs, dim=1)
        stats={prefix+"accuracy_"+name: errs.mean(i, dim=1), prefix+"std_"+name: std[i], prefix+"std_mean_"+name: std[i]/np.sqrt(samples*batch_size) for i, name in enumerate(model_names)}
        quantiles=torch.quantile(errs, torch.Tensor([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]), dim=1)
        for i, name in enumerate(model_names):
            stats[prefix+"max_"+name]=quantiles[i, len(quantiles)-1]
            stats[prefix+"min_"+name]=quantiles[i, 0]
            for j in range(1, len(quantiles)-1):
                stats[:, prefix+"quantile_"+name+str([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1][j])]=quantiles[i, j]
        if save_path!=None:
            os.makedirs(save_path, exist_ok=True)
            torch.save(stats, os.path.join(save_path,self.fn_cls.name))

        return stats

    def evaluate(self, models: Iterable[ContextModel]):# -> Iterable[Tensor]:

        samples=10 #should not be hardcoded, but unsure where this input should be.
        save_path=None
        
        sequence_length=self.fn_cls.sequence_length
        batch_size=self.fn_cls.batch_size
        errs=torch.zeros((len(models), samples, batch_size, sequence_length))

        for i, (x_batch, y_batch) in zip(range(samples), function_class):
            with torch.no_grad():
                errs[:, i] = torch.tensor([
                  
                    self._metric(
                        y_batch,
                        model.forward(x_batch, y_batch)
                    )
                    for model in models
                ])
        
        num_models, samples, batch_size, sequence_length=errs.size()

        errs=torch.reshape(errs, (num+models, samples*batch_size, sequence_length))

        return self.PostProcessingStats(errs, [model.name for model in models], save_path), errs

    def evaluateRobustness(self, models: Iterable[ContextModel]):
        samples=10 #should not be hardcoded, but unsure where this input should be.
        save_path=None
        
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
        errs=torch.zeros((len(models), samples, batch_size, sequence_length))
        

        for j, task in enumerate(robustness_tasks):
            errs=torch.zeros((samples, batch_size, seq_length))
    
            for i, (x_batch, y_batch) in zip(range(samples), function_class):

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
                    errs[:, i] = torch.tensor([
                  
                        self._metric(
                            curys,
                            model.forward(curxs, curys)
                        )
                        for model in models
                    ])

                output = model(curxs, curys)
        
                errs[i] = accuracy_func(output, curys)
        
            num_models, samples, batch_size, sequence_length=errs.size()

            errs=torch.reshape(errs, (num+models, samples*batch_size, sequence_length))

            robustness_nums.update(PostProcessingStats(errs, [model.name for model in models], save_path, task[0]+str(task[1])))
        
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
