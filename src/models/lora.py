from peft import LoraConfig, get_peft_model # pyright: ignore[reportPrivateImportUsage]
from typing import Optional, Any
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

        self.name = "lora_" + self.name

    def setup_peft_model(self, base_model_weights: Optional[Any] = False, lora_model_weights: Optional[Any] = False):

        assert not (base_model_weights and lora_model_weights), \
            f"at most one of base_model_weights and lora_model_weights can be passed into setup_peft_model"

        if not lora_model_weights:
            if base_model_weights:
                self.load_state_dict(base_model_weights)
            for param in self.parameters():
                param.requires_grad = False
            self._backbone = get_peft_model(self._backbone, self.lora_config) # pyright: ignore[reportArgumentType]
        elif lora_model_weights:
            self._backbone = get_peft_model(self._backbone, self.lora_config) # pyright: ignore[reportArgumentType]
            self.load_state_dict(lora_model_weights)
            for weight in lora_model_weights:
                if 'lora' not in weight:
                    lora_model_weights[weight].requires_grad = False
