import yaml
import torch

from train import TrainerSteps
from .trainer import get_value, expand_curriculum, get_x_distribution, get_function_class, get_optimizer, get_loss_fn, get_model

def parse_resume_training(content: str, latest_checkpoint_path: str, latest_step: int) -> TrainerSteps:
    data = yaml.load(content, Loader=yaml.Loader)['train']

    stages, step_counts = expand_curriculum(data)

    resume_idx, steps, step = 0, [], 0
    for step_count in step_counts:
        step += step_count
        steps.append(step)
    steps_past = list(filter(lambda s: s <= latest_step, steps))
    if not steps_past:
        resume_idx = 0
    else:
        resume_idx = steps.index(steps_past[-1])
    resume_stages, resume_step_counts = stages[resume_idx:], step_counts[resume_idx:]


    x_dim: int = max(
        get_value(data['x_dim'], data['steps']), # type: ignore
        get_value(data['x_dim'], 0), # type: ignore
    )

    _x_dist = get_x_distribution(
        stages[0]['b_size'], stages[0]['seq_len'], x_dim, stages[0].get('x_dist', {})
    )

    f_classes = [
        get_function_class(
            _x_dist,
            stage['x_dim'],
            stage['function_class']
        ) 
        for stage in resume_stages
    ]

    model = get_model(stages[0]['model'] | { "x_dim" : x_dim })
    optimizer = get_optimizer(model, stages[0]['optim'])

    latest_checkpoint = torch.load(latest_checkpoint_path)
    model.load_state_dict(latest_checkpoint['model_state_dict'])
    optimizer.load_state_dict(latest_checkpoint['optimizer_state_dict'])

    loss_fn = get_loss_fn(stages[0]['loss_fn'])
    baseline_models = list(map(
        lambda d: get_model(
            d | {"x_dim" : x_dim}
        ), 
        stages[0]['baseline_models']
    ))

    log_freq = stages[0].get('log_freq', -1)
    checkpoint_freq = stages[0].get('checkpoint_freq', -1)

    big_trainer = TrainerSteps(
        function_classes=f_classes,
        model=model,
        optim=optimizer, 
        loss_fn=loss_fn,
        steps=resume_step_counts,
        baseline_models=baseline_models,
        log_freq=log_freq,
        checkpoint_freq=checkpoint_freq,
        step_offset=latest_step,
    )

    return big_trainer