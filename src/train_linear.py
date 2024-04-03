from parse import parse_training
import wandb

wandb.init()

trainer, yaml_str = parse_training("train_linear.yml")
trainer.train()