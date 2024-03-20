from function_classes.linear import LinearRegression
from models.transformer import TransformerModel
from train.context_trainer import ContextTrainer

import torch 

DIM = 10
CONTEXT_LENGTH = 20
BATCH_SIZE = 64
shape = (BATCH_SIZE, CONTEXT_LENGTH, DIM)

x_distribution = torch.distributions.MultivariateNormal(torch.zeros(shape), torch.eye(DIM))
param_distribution = torch.distributions.normal.Normal(torch.zeros(), torch.)

# import code
# code.interact(local=locals())

function_class = LinearRegression(x_distribution=x_distribution, 
                                  param_distribution=param_distribution)
model = TransformerModel(n_dims=DIM,
                         n_positions=CONTEXT_LENGTH)
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = torch.nn.MSELoss()

context_trainer = ContextTrainer(function_class=function_class,
                                 model=model,
                                 optim=optim,
                                 loss_fn=loss_fn,
                                 num_steps=100000,
                                 log_freq=500)

context_trainer.train()