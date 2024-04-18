from peft import LoraConfig, get_peft_model # pyright: ignore[reportPrivateImportUsage]
import torch
from torch import nn

from .transformer import TransformerModel
from core import ContextModel

class Lora(TransformerModel):
    def __init__(self, base_model: TransformerModel, lora_config: dict, **kwargs):
        super(Lora, self).__init__(base_model._backbone, 
                                   base_model._n_dims, 
                                   base_model.context_length, # pyright: ignore[reportArgumentType]
                                   base_model.n_embd)

        self.lora_config = LoraConfig(**lora_config)

        for param in self.parameters():
            param.requires_grad = False
        self._backbone = get_peft_model(self._backbone, self.lora_config) # pyright: ignore[reportArgumentType]

        self.name = "lora_" + self.name
