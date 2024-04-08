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

def main(args: arg.Namespace):
    wandb.init()

    with open("conf/models.yml", 'r') as f:
        model_conf = f.read()
    with open(args.conffile, 'r') as f:
        train_conf = f.read()

    full_yaml = model_conf.strip() + '\n\n' \
                + nest_yaml("train", train_conf.strip())
    
    log_yaml(full_yaml)

    trainer = parse_training(full_yaml)

    trainer.train()


if __name__ == "__main__":
    parser = arg.ArgumentParser()

    parser.add_argument("--config", '-c', type=str, action='store', dest="conffile",
                        help="name of the config file to use for training")
    args = parser.parse_args()
    main(args)
