## Quickstart


Set up your environment with:
```
conda init zsh
conda env create -f environment.yaml
conda activate in-context-learning
```

Alternatively, you can instantiate a codespace [(docs)](https://docs.github.com/en/codespaces/getting-started/quickstart) that automagically does the above!


Run a training run specified by `<config_file>` with:
```
python src/ --config conf/train/<config_file>.yml
```

Resume a training run specified by `<config_file>` starting from `<run_id>/<checkpoint_file>` with:
```
python src/ --config conf/train/<config_file>.yml --resume models/<run_id>/<checkpoint_file>
```


## Extending this repo

We follow the following flow for objects in this repository:
![image of the flow map](<readme_assets/flow map.png>)

Initializing a `FunctionClass` or `ContextModel` passes the dictionary parsed directly from yaml as arguments. We recommend reading examples for your respective function class in [conf/](https://github.com/in-context-learning-2024/in-context/blob/main/conf/)

Make sure to initialize your parent class with `super().__init__(...)` in your `__init__` routine for your custom `FunctionClass` or `ContextModel`. Omitting this can result in unexpected (and currently unchecked) errors!

We try to make it easy to contribute by [adding models](#adding-models) or [function classes](#adding-function-classes):


---
### Adding Models

We wrap all sequence architectures and baselines as [`ContextModel`s](https://github.com/in-context-learning-2024/in-context/blob/main/src/core/context_model.py). The order of calls against a `ContextModel` is as follows:
- We `__init__` the `ContextModel` directly from the yaml configuration
- Inside the training loop and during evaluation, we call `evaluate` to generate predictions

`ContextModel`s thus have only two required methods:
- `__init__(...)`
- `evaluate(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor`

In addition, we require both `.context_length: int` and `.name: str` to be set for evaluation and visualization.

To use your context model, add an identifier to [`src/models/__init__.py`](https://github.com/in-context-learning-2024/in-context/blob/main/src/models/__init__.py) and refer to it as `type: <your identifier>` in yaml. Examples are in [`conf/include/models/`](https://github.com/in-context-learning-2024/in-context/tree/main/conf/include/models)


#### *Adding Hybrid blocks:*

We also provide an abstraction to compose hybrid architectures as a collection of *blocks*. The currently supported blocks are in [`src/models/hybrid.py`](https://github.com/in-context-learning-2024/in-context/blob/main/src/models/hybrid.py). To add a block, add an identifier to `SUPPORTED_BLOCKS` in [`src/models/hybrid.py`](https://github.com/in-context-learning-2024/in-context/blob/main/src/models/hybrid.py) and an entry to `MAPPING` in the function `SPEC_TO_MODULE` in the same file. To pass a value to the configuration fed into your new block, just specify it as a key/value pair in your `model:` mapping in YAML. Examples of this are in [`conf/include/models/composed.yml`](https://github.com/in-context-learning-2024/in-context/blob/main/conf/include/models/composed.yml), under `defs` -> `base`.


---
### Adding Function Classes

We take the approach of framing each function class as an iterator, where the boilerplate of making the iterator work for our training scheme (`ContextTrainer`) is [implemented for you](https://github.com/in-context-learning-2024/in-context/blob/main/src/core/function_class.py). The order of calls against a `FunctionClass` is as follows:
- We `__init__` the `FunctionClass` with the parsed yaml configuration
    - this calls `_init_param_dist()`, which should produce a [`torch.distributions.Distribution`](https://pytorch.org/docs/stable/distributions.html#distribution) 
        - Note: If [the distributions implemented in PyTorch](https://pytorch.org/docs/stable/distributions.html) are not enough, we also provide a way to ["combine" distributions](https://github.com/in-context-learning-2024/in-context/blob/main/src/utils.py)
- At the start of the training loop and/or evaluation, we produce an iterator with `__iter__` to sample batches from
- During the training loop and/or evaluation, we sample x,y batches with `__next__`
    - this calls `evaluate` to deterministically turn a sampled batch of x values and a sampled batch of parameters into a prediction

`FunctionClass`es thus have three methods you need to implement:
- `__init__(...)`
- `_init_param_dist(self)`
- `evaluate(self, x_batch: torch.Tensor, *params: torch.Tensor) -> torch.Tensor`

In addition, you need to export your function class in [`src/function_classes/__init__.py`](https://github.com/in-context-learning-2024/in-context/blob/main/src/function_classes/__init__.py)

To use your function class, you can specify the key you specified in [`src/function_classes/__init__.py`](https://github.com/in-context-learning-2024/in-context/blob/main/src/function_classes/__init__.py) as `type: <that key>` in a training yaml configuration. Examples are in [`conf/train/`](https://github.com/in-context-learning-2024/in-context/tree/main/conf/train)

