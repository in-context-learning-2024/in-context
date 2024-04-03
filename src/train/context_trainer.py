from core import FunctionClass
from torch.optim import Optimizer
from core import ContextModel
import torch
from torch import nn

import wandb
from typing import Optional, List, Any

class ContextTrainer:
    def __init__(
        self, 
        function_class: FunctionClass,
        model: ContextModel,
        optim: Optimizer, 
        loss_fn: nn.Module,
        steps: int,
        baseline_models: List,
        log_freq: int = -1,
        **kwargs
    ):
        self.function_class = function_class
        self.model = model
        self.optim = optim 
        self.loss_fn = loss_fn
        self.steps = steps
        self.baseline_models = baseline_models
        self.log_freq = log_freq 
        self.metadata = {k:v for k, v in zip(kwargs.keys(), kwargs.values()) if isinstance(v, (int, float))}

    def train(self, pbar: Optional[Any] = None) -> ContextModel:

        baseline_loss = {}

        for i, (x_batch, y_batch) in zip(range(self.steps), self.function_class):
            
            output = self.model(x_batch, y_batch)
            loss = self.loss_fn(output, y_batch)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f"loss {loss}")

            if i % self.log_freq == 0:

                for baseline in self.baseline_models:
                    baseline_output = baseline(x_batch, y_batch)
                    with torch.no_grad():
                        baseline_loss[baseline.name] = self.loss_fn(baseline_output, y_batch.cpu())

                log_dict = {
                        "overall_loss": loss,
                    } 
                log_dict |= {f"baseline_loss_{baseline.name}": baseline_loss[baseline.name] for baseline in self.baseline_models}

                wandb.log(
                    data=log_dict
                )

        return self.model

class TrainerSteps(ContextTrainer):

    def __init__(self, 
        function_classes: list[FunctionClass], 
        model: ContextModel, 
        optim: Optimizer, 
        loss_fn: nn.Module, 
        steps: list[int], 
        baseline_models: list[ContextModel],
        log_freq: int = -1,
        metadatas: Any = None
    ):

        assert len(function_classes) == len(steps), \
            f"The number of training stages does not match between step counts and function classes!"
        assert metadatas and len(metadatas) == len(function_classes), \
            f"Metadata for each function class is provided but does not match the number of function classes provided!"

        self.function_classes = function_classes
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.steps = steps
        self.baseline_models = baseline_models
        self.log_freq = log_freq
        self.metadatas = metadatas

        self.trainers = [
            ContextTrainer(
                fc,
                self.model,
                optim,
                loss_fn,
                step_count,
                baseline_model,
                log_freq
            )
            for fc, step_count, baseline_model in zip(function_classes, steps, baseline_models)
        ]

    def train(self, pbar: Optional[Any] = None) -> ContextModel:

        for fc, step_count, baseline_model, metadata in zip(self.function_classes, 
                                                            self.steps, 
                                                            self.baseline_models, 
                                                            self.metadatas if self.metadatas else [None] * len(self.fcs)):
            
            if self.metadatas:
                wandb.log(data=metadata, commit=False)

            trainer = ContextTrainer(
                fc,
                self.model,
                self.optim,
                self.loss_fn,
                step_count,
                baseline_model,
                self.log_freq
            )

            self.model = trainer.train(pbar)

        return self.model
