# pyright: reportMissingSuperCall=information

import torch
import wandb
import os
from torch import nn

from torch.optim import Optimizer
from typing import Optional, Any

from core import Baseline, TrainableModel, FunctionClass

class ContextTrainer:
    def __init__(
        self, 
        function_class: FunctionClass,
        model: TrainableModel,
        optim: Optimizer, 
        loss_fn: nn.Module,
        steps: int,
        baseline_models: list[Baseline],
        log_freq: int = -1,
        checkpoint_freq: int = -1,
        step_offset: int = 0,
        skip_steps: int = 0,
        predict_last: bool = False,
        **kwargs: Any, 
    ):
        super().__init__()
        self.function_class = function_class
        self.model = model
        self.optim = optim 
        self.loss_fn = loss_fn
        self.steps = steps
        self.baseline_models = baseline_models
        self.log_freq = log_freq 
        self.checkpoint_freq = checkpoint_freq
        self.step_offset = step_offset
        self.skip_steps = skip_steps
        self.predict_last = predict_last

    def _log(self, step: int, data: dict[str, Any]) -> None:
        global_step_num = step + self.step_offset
        wandb.log(
            data=data,
            step=global_step_num,
            commit=True
        )

    def _checkpoint(self, step: int) -> None:
        global_step_num = step + self.step_offset
        if self.checkpoint_freq > 0 and global_step_num % self.checkpoint_freq == 0:
            checkpoint = {'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optim.state_dict()}

            # save locally
            local_dir_path = f"models/{os.path.basename(os.path.dirname(wandb.run.dir)).replace('run-', '')}" # pyright: ignore [reportOptionalMemberAccess]
            os.makedirs(local_dir_path, exist_ok=True)
            torch.save(checkpoint, os.path.join(local_dir_path, f"checkpoint_{global_step_num}"))

            # save in wandb
            wandb_dir_path = os.path.join(wandb.run.dir, 'models') # pyright: ignore [reportOptionalMemberAccess]
            wandb_path = os.path.join(wandb_dir_path, f"checkpoint_{global_step_num}")
            os.makedirs(wandb_dir_path, exist_ok=True)
            torch.save(checkpoint, wandb_path)
            wandb.save(wandb_path, base_path=wandb.run.dir) # pyright: ignore [reportOptionalMemberAccess]

    def train(self, pbar: Optional[Any] = None) -> TrainableModel:

        baseline_loss = {}

        for i, (x_batch, y_batch) in zip(range(self.skip_steps, self.steps), self.function_class):

            output = self.model.evaluate(x_batch, y_batch)
            if output.shape != y_batch.shape:
                raise ValueError(
                    f"Model {self.model.name} produced ill-shaped predictions!"
                    + f"Expected: {y_batch.shape}    Got: {output.shape}"
                )

            if not self.predict_last:
                loss = self.loss_fn(output, y_batch)
            else:
                loss = self.loss_fn(output[:, -1], y_batch[:, -1])

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f"loss {loss}")

            if self.log_freq > 0 and i % self.log_freq == 0:
                for baseline in self.baseline_models:
                    baseline_output = baseline.evaluate(x_batch, y_batch)

                    if baseline_output.shape != y_batch.shape:
                        raise ValueError(
                            f"Baseline model {baseline.name} produced ill-shaped predictions!" + \
                                f"Expected: {y_batch.shape}    Got: {baseline_output.shape}"
                        )

                    baseline_loss[baseline.name] = self.loss_fn(baseline_output.cpu(), y_batch.cpu()).detach()

                log_dict = {
                    "overall_loss": loss,
                }
                log_dict |= {
                    f"baseline_loss_{baseline.name}": baseline_loss[baseline.name] 
                    for baseline in self.baseline_models
                }

                self._log(step=i, data=log_dict)

            self._checkpoint(step=i)

        return self.model

class TrainerSteps(ContextTrainer):

    def __init__(self,
        function_classes: list[FunctionClass], 
        model: TrainableModel, 
        optim: Optimizer, 
        loss_fn: nn.Module, 
        steps: list[int], 
        baseline_models: list[Baseline],
        log_freq: int = -1,
        checkpoint_freq: int = -1,
        skip_steps: int = 0,
        predict_last: bool = False,
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
        self.skip_steps_left = skip_steps
        self.step_offset = 0
        self.predict_last = predict_last

    def train(self, pbar: Optional[Any] = None) -> TrainableModel:

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
                self.step_offset,
                self.skip_steps_left,
                self.predict_last,
            )

            self.model = trainer.train(pbar)

            self.step_offset += step_count
            self.skip_steps_left = max(0, self.skip_steps_left - step_count)

        return self.model
