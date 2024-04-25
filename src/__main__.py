import argparse as arg
import wandb
import torch
import os
import yaml

from typing import Optional

from parse import parse_training

def nest_yaml(tag: str, content: str, indent_size:int=4) -> str:
    indent = indent_size * ' '
    
    lines = content.split('\n')
    lines = [ indent + line for line in lines ]

    return f"{tag}:\n" + "\n".join(lines)

def log_yaml(full_yaml: str) -> None:
    # save locally
    local_dir_path = f"models/{os.path.basename(os.path.dirname(wandb.run.dir)).replace('run-', '')}" # pyright: ignore [reportOptionalMemberAccess]
    local_config_path = os.path.join(local_dir_path, "config.yml")
    os.makedirs(local_dir_path, exist_ok=True)
    with open(local_config_path, 'w') as f:
        f.write(full_yaml)

    # save in wandb
    wandb_dir_path = os.path.join(wandb.run.dir, "conf/") # pyright: ignore [reportOptionalMemberAccess]
    wandb_conf_path = os.path.join(wandb_dir_path, "config.yml")
    os.makedirs(wandb_dir_path, exist_ok=True)
    with open(wandb_conf_path, 'w') as f:
        f.write(full_yaml)
    wandb.save(wandb_conf_path, base_path=wandb.run.dir) # pyright: ignore [reportOptionalMemberAccess]

def load_config(conffile: str) -> str:
    with open("conf/models.yml", 'r') as f:
        model_conf = f.read()
    with open(conffile, 'r') as f:
        train_conf = f.read()

    full_yaml = model_conf.strip() + '\n\n' \
                + nest_yaml("train", train_conf.strip())

    return full_yaml

def load_checkpoint(checkpointfile: str) -> tuple:
    latest_checkpoint = torch.load(checkpointfile)
    return latest_checkpoint['model_state_dict'], latest_checkpoint['optimizer_state_dict']

def main(args: arg.Namespace):

    yaml_str = load_config(args.conffile)

    init_args = { "config" : yaml.load(yaml_str, Loader=yaml.Loader) } 
    if args.projectname is not "":
        init_args |= { "project" : args.projectname }
    if args.runname is not "":
        init_args |= { "name" : args.runname }
    wandb.init(**init_args)

    log_yaml(yaml_str)

    if args.checkpointfile == "":
        trainer = parse_training(yaml_str)
    else:
        latest_step = int(os.path.basename(args.checkpointfile).split("_")[-1])
        model_weights, optim_state = load_checkpoint(args.checkpointfile)

        trainer = parse_training(
            yaml_str,
            skip_steps=latest_step,
            model_weights=model_weights,
            optim_state=optim_state
        )

    trainer.train()


if __name__ == "__main__":
    parser = arg.ArgumentParser()

    parser.add_argument("--config", '-c', type=str, default="", action='store', dest="conffile",
                        help="path to the config file to use for training")
    parser.add_argument("--resume", '-r', type=str, default="", action='store', dest="checkpointfile",
                        help="path to the checkpoint to resume training with")
    parser.add_argument("--wandb-project", type=str, default="", dest="projectname")
    parser.add_argument("--run-name", type=str, default="", dest="runname")
    args = parser.parse_args()
    main(args)
