import argparse as arg
import wandb
import os

from parse import parse_training

def nest_yaml(tag: str, content: str, indent_size:int=4) -> str:
    indent = indent_size * ' '
    
    lines = content.split('\n')
    lines = [ indent + line for line in lines ]

    return f"{tag}:\n" + "\n".join(lines)

def log_yaml(full_yaml: str) -> None:
    # save locally
    local_dir_path = f"models/{os.path.basename(os.path.dirname(wandb.run.dir)).replace('run-', '')}"
    local_config_path = os.path.join(local_dir_path, "config.yml")
    os.makedirs(local_dir_path, exist_ok=True)
    with open(local_config_path, 'w') as f:
        f.write(full_yaml)

    # save in wandb
    wandb_dir_path = os.path.join(wandb.run.dir, "conf/")
    wandb_conf_path = os.path.join(wandb_dir_path, "config.yml")
    os.makedirs(wandb_dir_path, exist_ok=True)
    with open(wandb_conf_path, 'w') as f:
        f.write(full_yaml)
    wandb.save(wandb_conf_path, base_path=wandb.run.dir)

def train_from_scratch(args: arg.Namespace):
    with open("conf/models.yml", 'r') as f:
        model_conf = f.read()
    with open(args.conffile, 'r') as f:
        train_conf = f.read()

    full_yaml = model_conf.strip() + '\n\n' \
                + nest_yaml("train", train_conf.strip())
    log_yaml(full_yaml)

    trainer = parse_training(full_yaml)
    trainer.train()

def resume_training(args):
    train_dir = os.path.join("models/", args.resumetrainid)
    with open(os.path.join(train_dir, "full_yaml.yml"), 'r') as f:
        full_yaml = f.read()
    checkpoints = list(filter(lambda f: f != "full_yaml.yml", os.listdir(train_dir)))
    latest_checkpoint = sorted(checkpoints, key=lambda f: int(f.split('_')[-1]))[-1]
    latest_step = int(latest_checkpoint.split('_')[-1])

    # TODO: use latest_step to make a new full_yaml file that has current the curriculum information, also has checkpointed model as the initial model
    full_yaml = ...
    log_yaml(full_yaml)

    # TODO: use latest_checkpoint to reinstantiate the model, need another way of creating contextmodels
    trainer = parse_training(full_yaml)
    trainer.train()
    
    print(latest_checkpoint)
    print(latest_step)

def main(args: arg.Namespace):
    wandb.init()

    if args.conffile:
        train_from_scratch(args)
    elif args.resumetrainid:
        resume_training(args)
    else:
        raise AssertionError(f"Invalid Arguments: {args}")


if __name__ == "__main__":
    parser = arg.ArgumentParser()

    parser.add_argument("--config", '-c', type=str, default="", action='store', dest="conffile",
                        help="name of the config file to use for training")
    parser.add_argument("--resume", '-r', type=str, default="", action='store', dest="resumetrainid",
                        help="id of the training run to resume training of latest checkpoint")
    args = parser.parse_args()
    main(args)
