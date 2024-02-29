from transformer import TransformerModel

def throw(ex):
    raise ex

def build_model(conf):
    cls = {
        "gpt2" : TransformerModel,
        # "relu_attn" : TransformerRelu,
        # "relu_attn_causal" : TransformerReluCausal,
        # "mlp" : MLPSequence,
    }.get(
        conf.family,
        lambda *_, **__: throw(
            NotImplementedError(f"Invalid model family!: '{conf.family}'")
        )
    )

    model = cls(
        n_dims=conf.n_dims,
        n_positions=conf.n_positions,
        n_embd=conf.n_embd,
        n_layer=conf.n_layer,
        n_head=conf.n_head,
        **conf.kwargs
    )

    return model

def get_relevant_baselines(task_name):
    task_to_baselines = {
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_classification": [
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models

