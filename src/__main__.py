import argparse as arg
import wandb

from parse import parse_training

def nest_yaml(tag: str, content: str, indent_size:int=4) -> str:
    indent = indent_size * ' '
    
    lines = content.split('\n')
    lines = [ indent + line for line in lines ]

    return f"{tag}:\n" + "\n".join(lines)

def main(args: arg.Namespace):
    wandb.init()

    with open("conf/models.yml", 'r') as f:
        model_conf = f.read()
    with open(args.conffile, 'r') as f:
        train_conf = f.read()

    full_yaml = model_conf.strip() + '\n\n' \
                + nest_yaml("train", train_conf.strip())

    trainer = parse_training(full_yaml)

    print(full_yaml)
    trainer.train()


if __name__ == "__main__":
    parser = arg.ArgumentParser()

    parser.add_argument("--config-file", action='store', dest="conffile")
    args = parser.parse_args()
    main(args)
