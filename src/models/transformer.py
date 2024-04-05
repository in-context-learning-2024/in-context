import torch
from transformers import GPT2Config, GPT2Model, LlamaConfig, LlamaModel
from torch import nn

from core import ContextModel

ARCHITECTURES = {
    "gpt2": {'config': GPT2Config, 'model': GPT2Model},
    "llama2": {'config': LlamaConfig, 'model': LlamaModel}
} 
class TransformerModel(ContextModel):
    def __init__(self, x_dim, n_positions, n_embd=128, n_layer=12, n_head=4, **kwargs):
        super(TransformerModel, self).__init__()
        architecture = ARCHITECTURES[kwargs["architecture"]]
        configuration = architecture['config'](
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.context_length = n_positions
        self._n_dims = x_dim
        self._read_in = nn.Linear(x_dim, n_embd)
        self._backbone = architecture['model'](configuration)
        self._read_out = nn.Linear(n_embd, 1)

    def forward(self, xs, ys):
        self._backbone.to(xs.device) # type: ignore
        inds = torch.arange(ys.shape[1])

        zs = ContextModel.interleave(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state # type: ignore
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs
