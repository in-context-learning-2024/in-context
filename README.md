## Quickstart


Set up your environment with:
```
conda init zsh
conda env create -f environment.yaml
conda activate in-context-learning
```


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

These initialized classes are passed the dictionary parsed directly from yaml as arguments. We recommend reading examples for your respective function class in [conf/](https://github.com/in-context-learning-2024/in-context/blob/main/conf/)

Make sure to initialize your parent class with `super().__init__(...)` in both cases

We try to make it easy to contribute by [adding models](#adding-models) or [function classes](#adding-function-classes):


---
### Adding Models

We wrap all sequence architectures and baselines as [`ContextModel`s](https://github.com/in-context-learning-2024/in-context/blob/main/src/core/context_model.py). The order of calls against a `ContextModel` is as follows:
- We `__init__` the `ContextModel` directly from the yaml configuration
- Inside the training loop and during evaluation, we call `forward` to generate predictions

`ContextModel`s thus have only two required methods:
- `__init__(...)`, and
- `forward(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor`

In addition, we require both `.context_length: int` and `.name: str` to be set for evaluation and visualization.

---
### Adding Function Classes

We take the approach of framing each function class as an iterator, where the boilerplate of making the iterator work for our training scheme (`ContextTrainer`) is [implemented for you](https://github.com/in-context-learning-2024/in-context/blob/main/src/core/function_class.py). The order of calls against a `FunctionClass` is as follows:
- We `__init__` the `FunctionClass` with the parsed yaml configuration
    - this calls `_init_param_dist()`, which should produce a [`torch.distributions.Distribution`](https://pytorch.org/docs/stable/distributions.html#distribution) 
        - Note: If the implemented ones in pytorch are not enough, we also provide a way to ["combine" distributions](https://github.com/in-context-learning-2024/in-context/blob/main/src/utils.py)
- At the start of the training loop and/or evaluation, we produce an iterator with `__iter__` to sample batches from
- During the training loop and/or evaluation, we sample x,y batches with `__next__`
    - this calls `evaluate` to deterministically turn a sampled batch of x values and a sampled batch of parameters into a prediction

`FunctionClass`es thus have three methods you need to implement:
- `__init__(...)`
- `_init_param_dist(self)`
- `evaluate(self, x_batch: torch.Tensor, *params: torch.Tensor) -> torch.Tensor`
