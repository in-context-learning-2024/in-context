import torch

from torch import Tensor
from typing import Iterable

from tqdm import tqdm

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

    def evaluate(self, models: Iterable[ContextModel], num_batches: int = 1, perfect_model: ContextModel =None) -> Iterable[Tensor]:
        """Produce a tensor of shape (batch_size * num_batches, metric_shape) for each model provided"""

        if perfect_model!=None:
            batch_size = self.function_class.batch_size
            sequence_length = self.function_class.sequence_length
            y_dim = self.function_class.y_dim
            models = list(models)
            num_models = len(models)
            metric_dim=y_dim*2
            errs = torch.zeros((num_batches,num_models, batch_size, sequence_length , metric_dim))
            for batch_num, (x_batch, y_batch) in tqdm(zip(range(num_batches), self.function_class)):
                perfect_pred=perfect_model.forward(x_batch, y_batch)
                for model_num, model in enumerate(models):
                    with torch.no_grad():
                        y_pred=model.forward(x_batch, y_batch)
                        errs[batch_num, model_num, :, :, :y_dim]=self.metric.evaluate(y_batch, y_pred)
                        errs[batch_num, model_num, :, :, y_dim:]=self.metric.evaluate(perfect_pred, y_pred)

        else:
            with torch.no_grad():
                errs = torch.stack([
                    torch.stack([
                        self.metric.evaluate(
                            y_batch,
                            model.forward(x_batch, y_batch)
                        )
                        for model in models
                    ])  
                    for _, (x_batch, y_batch) in tqdm(zip(range(num_batches), self.function_class))
                ])# errs is of shape: (#batches, #models, batch_size, sequence_length, *metric_dims)

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

    def evaluate(self, models: Iterable[ContextModel], num_batches: int = 1, perfect_model: ContextModel =None) -> Iterable[Tensor]:

        batch_size = self.function_class.batch_size
        sequence_length = self.function_class.sequence_length
        y_dim = self.function_class.y_dim
        models = list(models)
        num_models = len(models)

        metric_dim=y_dim

        if perfect_model!=None:
            metric_dim=2*y_dim

        errs = torch.zeros((num_batches, num_models, batch_size, sequence_length, metric_dim))
        for batch_num in tqdm(range(num_batches)):
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
                if perfect_model==None:
                    with torch.no_grad():
                        errs[batch_num, :, :, index, :] = torch.stack([
                            self.metric.evaluate(
                                y_query, # shape (batch_size, y_dim)
                                model.forward(x_comb, ys_context[:, :index])[:, -1] # shape (batch_size, y_dim)
                            ) for model in models
                        ])
                else:
                    with torch.no_grad():
                        perfect_pred=perfect_model.forward(x_comb, ys_context[:, :index])[:, -1]
                        for model_num, model in enumerate(models):
                            y_pred=model.forward(x_comb, ys_context[:, :index])
                            errs[batch_num, model_num, :, index, :y_dim]=self.metric.evaluate(y_query, # shape (batch_size, y_dim)
                                y_pred[:, -1]) # shape (batch_size, y_dim)
                            errs[batch_num, model_num, :, index, y_dim:]=self.metric.evaluate(perfect_pred, # shape (batch_size, y_dim)
                                y_pred[:, -1]) # shape (batch_size, y_dim)
                            

        errs = torch.transpose(errs, 0, 1) # shape (num_models, #batches, batch_size,, seq_len, metric_dim)
        errs = torch.flatten(errs, 1, 2)

        return errs


class FCErrorOrthogonal(FunctionClassError):

    def __init__(self, metric: Metric, function_class: FunctionClass, rescale=True):
        super(FCErrorOrthogonal, self).__init__(metric, function_class)
        self.rescale=rescale

    def evaluate(self, models: Iterable[ContextModel], num_batches: int = 1,  perfect_model: ContextModel =None) -> Iterable[Tensor]:
        sequence_length = self.function_class.sequence_length
        batch_size = self.function_class.batch_size
        y_dim = self.function_class.y_dim
        num_models = len(list(models))

        metric_dim=y_dim
        if perfect_model!=None:
            metric_dim=2*y_dim

        errs = torch.zeros((num_models, num_batches, batch_size, sequence_length, metric_dim))

        for batch_num in tqdm(range(num_batches)):
            params = self.function_class.p_dist.sample()
            x_batch = self.function_class.x_dist.sample()
            n = x_batch.shape[2]
            
            A = torch.randn(batch_size, n, n)
            Q, _ = torch.linalg.qr(A, mode="complete")
            context_space = Q.clone()
            context_space[:, :, -1] = 0
            context_space= context_space @ torch.transpose(context_space, dim0=1, dim1=2)
            test_space = Q.clone()
            test_space[:, :, :-1] = 0
            test_space= test_space @torch.transpose(test_space, dim0=1, dim1=2)
            
            x_context= x_batch @context_space
            x_test =   x_batch @test_space
            
            if self.rescale:
                x_context=x_context*x_batch.norm(dim=2).unsqueeze(2)/x_context.norm(dim=2).unsqueeze(2)
                
                x_test=x_test*x_batch.norm(dim=2).unsqueeze(2)/x_test.norm(dim=2).unsqueeze(2)

            for index in range(sequence_length):
                
                cur_x = x_context.clone()
                cur_x[:,index] = x_test[:, index]

                if isinstance(params, list):
                    y_test = self.function_class.evaluate(cur_x, *params)
                else:
                    y_test = self.function_class.evaluate(cur_x, params)
                if perfect_model==None:
                    with torch.no_grad():
                        errs[:, batch_num, :, index] = torch.stack([
                            self.metric.evaluate(
                                y_test,
                                model.forward(cur_x, y_test)
                            )
                            for model in models
                        ])[:, :, index]
                else:
                    with torch.no_grad():
                        perfect_pred=perfect_model.forward(cur_x, y_test)
                        for model_num, model in enumerate(models):
                            y_pred=model.forward(cur_x, y_test)
                            errs[model_num, batch_num, :, index, :y_dim]=self.metric.evaluate(y_test, 
                                y_pred)[:, index] 
                            errs[model_num, batch_num, :, index, y_dim:]=self.metric.evaluate(perfect_pred, 
                                y_pred)[:, index]
                
        
        errs = torch.reshape(errs, (num_models, num_batches*batch_size, sequence_length, metric_dim))

        return errs


class FCErrorSeenPoints(FunctionClassError):

    def evaluate(self, models: Iterable[ContextModel], num_batches: int = 1, perfect_model: ContextModel=None) -> Iterable[Tensor]:
        sequence_length = self.function_class.sequence_length
        batch_size = self.function_class.batch_size
        y_dim = self.function_class.y_dim
        num_models = len(list(models))
 
        metric_dim=y_dim
        if perfect_model!=None:
            metric_dim=2*y_dim

        errs = torch.zeros((num_models, num_batches, batch_size, sequence_length, metric_dim))


        for batch_num in tqdm(range(num_batches)):
            params = self.function_class.p_dist.sample()
            x_batch = self.function_class.x_dist.sample()
            
            for index in range(1, sequence_length):
                x_test = x_batch.clone()
                perm = torch.stack([torch.randperm(index) for _ in range(batch_size)]).unsqueeze(dim=1) #samples a permutation from 0 to index-1 for each sequence in the batch
                ind_mat = (perm == 0) + 0.0 #creates a tensor that is one where the zero is in each permutation, and zero everywhere else
                x_test[:, index:index+1] = ind_mat @ x_batch[:, :index] #takes the x-values from the index of x_batch corresponding to the 1 of ind_mat

                if isinstance(params, list):
                    y_test = self.function_class.evaluate(x_test, *params)
                else:
                    y_test = self.function_class.evaluate(x_test, params)
                
                if perfect_model==None:
                    with torch.no_grad():
                        errs[:, batch_num, :, index] = torch.stack([
                  
                            self.metric.evaluate(
                                y_test,
                                model.forward(x_test, y_test)
                            )
                            for model in models
                        ])[:, :, index]
                else:
                    with torch.no_grad():
                        perfect_pred=perfect_model.forward(x_test, y_test)
                        for model_num, model in enumerate(models):
                            y_pred=model.forward(x_test, y_test) 
                            errs[model_num, batch_num, :, index, :y_dim] =self.metric.evaluate(y_test,
                                    y_pred)[:, index]
                            errs[model_num, batch_num, :, index, y_dim:] =self.metric.evaluate(perfect_pred,
                                    y_pred)[:, index]
        
        errs = torch.reshape(errs, (num_models, num_batches*batch_size, sequence_length, metric_dim))[:, :, 1:]

        return errs
