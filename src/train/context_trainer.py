import os
from tqdm import tqdm
import torch
import yaml
import wandb

class ContextTrainer(Serializable):
    def __init__(self, **config):
        # TODO: use config here
        self.optimizer = config.get("optimizer") # torch.optim.Adam(config)
        self.loss_func = ... # config.get("loss_func")
        self.func_class = ...
        self.metadata = ... # config stuff here
        pass

    def step_train(self, model, ...) -> None:
        # TODO: optimize the next three lines
        seq_batch = [[(xs, ys) for xs, ys in function] for function in self.func_class.get_function_iter()]
        xs = [[xy_pair[0] for xy_pair in seq] for seq in seq_batch]
        ys = [[xy_pair[1] for xy_pair in seq] for seq in seq_batch]

        self.optimizer.zero_grad()
        output = model(xs, ys)
        loss = self.loss_func(output, ys)
        loss.backward()
        self.optimizer.step()

        # TODO: log the loss
        # TODO: log the xs, ys? functions?

    def train(self, model, steps, ...) -> None:
        for _ in tqdm(range(steps)):
            self.step_train(model)
            # TODO: curriculum?

    @property
    def metadata(self) -> dict:
        return self.metadata