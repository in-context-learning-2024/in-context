import argparse as arg
import wandb
import os
import yaml

from typing import Any

from parse import parse_training_from_file

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

    if args.checkpointfile == "":
        trainer, config = parse_training_from_file(
            filename=args.conffile,
            include=args.includedir
        )
    else:
        trainer, config = parse_training_from_file(
            filename=args.conffile,
            checkpoint_path=args.checkpointfile,
            include=args.includedir
        )

    init_args: dict[str, Any] = { "config" : config }
    if args.projectname != "":
        init_args |= { "project" : args.projectname }
    if args.runname != "":
        init_args |= { "name" : args.runname }
    wandb.init(**init_args)

    log_yaml(yaml.dump(config, Dumper=yaml.Dumper))

    trainer.train()


if __name__ == "__main__":
    conf_include_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "conf", "include")
    parser = arg.ArgumentParser()

    parser.add_argument("--config", '-c', type=str, default="", action='store', dest="conffile",
                        help="path to the config file to use for training")
    parser.add_argument("--resume", '-r', type=str, default="", action='store', dest="checkpointfile",
                        help="path to the checkpoint to resume training with")
    parser.add_argument("--wandb-project", type=str, default="", dest="projectname",
                        help="the project to log to in weights and biases")
    parser.add_argument("--run-name", type=str, default="", dest="runname",
                        help="what to name this run in wandb")
    parser.add_argument("--include-dir", type=str, default=conf_include_dir, action='store', dest="includedir",
                        help="from what directory to include configurations")
    args = parser.parse_args()
    main(args)
