from function_classes.function_class import FunctionClass
from torch.optim import Optimizer
from models.context_model import ContextModel
from torch import nn

# import wandb
from typing import Optional, Any

class ContextTrainer:
    def __init__(
        self, 
        function_class: FunctionClass,
        model: ContextModel,
        optim: Optimizer, 
        loss_fn: nn.Module,
        num_steps: int,
        log_freq: int
    ):
        self.func_class = function_class
        self.model = model
        self.optimizer = optim 
        self.loss_func = loss_fn
        self.num_steps = num_steps
        self.log_freq = log_freq 
        #self.metadata = ... # config stuff here ********* TODO: excuse me what
        # wandb.log(self.metadata)


    def train(self, pbar: Optional[Any] = None) -> ContextModel:

        for i, (x_batch, y_batch) in zip(range(self.num_steps), self.func_class):
            
            output = self.model(x_batch, y_batch)
            loss = self.loss_func(output, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f"loss {loss}")
            # TODO: curriculum?

            # TODO: stretch goal: log functions / precompute dataset?

            # if i % self.log_freq == 0:
            #     wandb.log(
            #         {
            #             "overall_loss": loss,
            #             # TODO: log the xs, ys? 
            #         },
            #         step=i,
            #     )

        return self.model

    @property
    def metadata(self) -> dict:
        return self.metadata

class TrainerSteps(ContextTrainer):

    def __init__(self, 
        function_classes: list[FunctionClass], 
        model: ContextModel, 
        optim: Optimizer, 
        loss_fn: nn.Module, 
        num_steps: list[int], 
        log_freq: int
    ):

        assert len(function_classes) == len(num_steps), \
            f"The number of training stages does not match between step counts and function classes!"

        self.fcs = function_classes
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.num_steps = num_steps
        self.log_freq = log_freq

        self.trainers = [
            ContextTrainer(
                fc,
                model,
                optim,
                loss_fn,
                step_count,
                log_freq
            )
            for fc, step_count in zip(function_classes, num_steps)
        ]

    def train(self, pbar: Optional[Any] = None) -> ContextModel:

        for fc, step_count in zip(self.fcs, self.num_steps):
            trainer = ContextTrainer(
                fc,
                self.model,
                self.optim,
                self.loss_fn,
                step_count,
                self.log_freq
            )
            self.model = trainer.train(pbar)

        return self.model
