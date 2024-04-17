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
