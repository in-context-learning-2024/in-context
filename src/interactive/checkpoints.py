
from core import ContextModel
from parse import parse_training_from_file

def load_checkpoint_with_config(config_file: str, checkpoint_file: str) -> ContextModel:
    trainer, _ = parse_training_from_file(config_file, checkpoint_file)
    return trainer.model
