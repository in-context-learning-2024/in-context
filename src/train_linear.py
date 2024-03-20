from function_classes.linear import LinearRegression
from models.transformer import TransformerModel
from train.context_trainer import ContextTrainer
from parse import parse_elaborated_stages

import torch 
import wandb

wandb.init()

# TODO: (potential idea) have parser JUST return model + list of context trainers
stages, step_counts = parse_elaborated_stages("train_linear.yml") # TODO: is this function working correctly, or do I not understand what it's supposed to be doing?

model = stages[0]['train']['model']
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = torch.nn.MSELoss()

for stage, step_count in zip(stages, step_counts):

    context_trainer = ContextTrainer(function_class=stage['train']['function_class'],
                                     model=model,
                                     optim=optim,
                                     loss_fn=loss_fn,
                                     steps=step_count,
                                     baseline_models=stage['train']['baseline_models'],
                                     log_freq=-1)
    
    curriculum_dict = {'seq_len': stage['train']['seq_len'], 'x_dim': stage['train']['x_dim']}
    wandb.log(curriculum_dict)

    context_trainer.train()