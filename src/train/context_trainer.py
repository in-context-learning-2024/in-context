from core import FunctionClass
from torch.optim import Optimizer
from core import ContextModel
import torch
from torch import nn

import wandb
import os
from typing import Optional, List, Any

class ContextTrainer:
    def __init__(
        self, 
        function_class: FunctionClass,
        model: ContextModel,
        optim: Optimizer, 
        loss_fn: nn.Module,
        steps: int,
        baseline_models: List[ContextModel],
        log_freq: int = -1,
        checkpoint_freq: int = -1,
        step_offset: int = 0,
        **kwargs
    ):
        self.function_class = function_class
        self.model = model
        self.optim = optim 
        self.loss_fn = loss_fn
        self.steps = steps
        self.baseline_models = baseline_models
        self.log_freq = log_freq 
        self.checkpoint_freq = checkpoint_freq
        self.step_offset = step_offset

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

            if self.log_freq > 0 and i % self.log_freq == 0:

                with torch.no_grad():
                    for baseline in self.baseline_models:
                        baseline_output = baseline(x_batch, y_batch)
                        baseline_loss[baseline.name] = self.loss_fn(baseline_output.cpu(), y_batch.cpu())

                log_dict = {
                        "overall_loss": loss,
                    } 
                log_dict |= {f"baseline_loss_{baseline.name}": baseline_loss[baseline.name] for baseline in self.baseline_models}

                wandb.log(
                    data=log_dict,
                    step=i + self.step_offset,
                    commit=True
                )
            
            if self.checkpoint_freq > 0 and (i + self.step_offset) % self.checkpoint_freq == 0:

                # save locally
                local_dir_path = f"models/{os.path.basename(os.path.dirname(wandb.run.dir)).replace('run-', '')}"
                if not os.path.exists(local_dir_path):
                    os.makedirs(local_dir_path)
                torch.save(self.model.state_dict(), os.path.join(local_dir_path, f"checkpoint_{i + self.step_offset}"))

                # save in wandb
                wandb_dir_path = os.path.join(wandb.run.dir, 'models')
                if not os.path.exists(wandb_dir_path):
                    os.makedirs(wandb_dir_path)
                wandb_path = os.path.join(wandb_dir_path, f"checkpoint_{i + self.step_offset}")
                torch.save(self.model.state_dict(), wandb_path)
                wandb.save(wandb_path, base_path=wandb.run.dir)

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
        checkpoint_freq: int = -1,
    ):

        assert len(function_classes) == len(steps), \
            f"The number of training stages does not match between step counts and function classes!"
        
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
        self.checkpoint_freq = checkpoint_freq

        self.trainers = [
            ContextTrainer(
                fc,
                self.model,
                optim,
                loss_fn,
                step_count,
                self.baseline_models,
                log_freq
            )
            for fc, step_count, in zip(function_classes, steps)
        ]

    def train(self, pbar: Optional[Any] = None) -> ContextModel:

        global_step = 0

        for fc, step_count, in zip(self.function_classes, self.steps):

            trainer = ContextTrainer(
                fc,
                self.model,
                self.optim,
                self.loss_fn,
                step_count,
                self.baseline_models,
                self.log_freq,
                self.checkpoint_freq,
                global_step
            )

            self.model = trainer.train(pbar)

            global_step += step_count

        return self.model
