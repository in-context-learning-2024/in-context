import wandb
import argparse

from parse import parse_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='read the name of the config files')
    
    parser.add_argument('--config', '-c', type=str, help='name of the config file to use for training')

    args = vars(parser.parse_args())
    yml_path = args['config']

    wandb.init()

    trainer, yaml_str = parse_training(yml_path)
    trainer.train()