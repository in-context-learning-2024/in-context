import torch

from typing import Iterable
from torch import Tensor
from tqdm import tqdm

import matplotlib.pyplot as plt

from scipy.stats import norm

def post_process(results: Iterable[Tensor],
                 confidence_level: list[float] = [0.99, 0.95],
                 quantile_cutoffs: list[float] = [0.05, 0.25, 0.5, 0.75, 0.95],
                ) -> list[dict[str, Tensor]]:
    summaries: list[dict[str, Tensor]] = [ ]
    for err_tensor in results:
        err_tensor=err_tensor.cpu()
        sample_count, *_ = err_tensor.size()

        QUANTILES = [0, *quantile_cutoffs, 1]
        std = torch.std(err_tensor, dim=0)
        std_err_of_mean = std / (sample_count ** 0.5)
        mean = torch.mean(err_tensor, dim=0)
        quantiles = torch.quantile(
            err_tensor, torch.tensor(QUANTILES), dim=0
        )

        confidence_data = { }
        for level in confidence_level:
            interval_jump = norm.ppf((1+level)/2) * std_err_of_mean
            confidence_data[f"confidence_{level}_upper"] = mean + interval_jump
            confidence_data[f"confidence_{level}_lower"] = mean - interval_jump

        summaries.append({
            f"accuracy" : mean,
            f"std" : std,
            f"std_mean" : std_err_of_mean,
            f"max" : quantiles[len(quantiles)-1],
            f"min" : quantiles[0],
            **confidence_data,
            **{
                f"quantile_{q_interval}" : q_value
                for q_interval, q_value in zip(QUANTILES[1:-1], quantiles[1:-1])
            },
        })
    return summaries

def average_evals(MODELS, Benchmark, num_batches, num_rounds, perfect_model=None): #runs multiple rounds of evalutions and averages them to save memory

    errs=Benchmark.evaluate(MODELS, num_batches, perfect_model=perfect_model)
    for i in tqdm(range(1, num_rounds)):
        errs=(i*errs+Benchmark.evaluate(MODELS, num_batches, perfect_model=perfect_model))/(i+1)
    return errs

def plot_comparison_2_models(MODELS=None, names=None,errs=None, post_processed_errs=None, post_processed_diff=None, title=None, confidence_level=[0.95, 0.99], ylim=None): 
    #compares the two first models in values or errs
    #if diff or errs is included the difference in their losses is also plotted
    if post_processed_errs ==None and errs==None:
        print("Needs either post_processed_errs or errs to not be None")
        return 
    if names==None and MODELS ==None:
        print("Plotting function needs either names or the models themselves")
        return 
    if names==None:
        names=[model.name for model in MODELS]

    if post_processed_errs==None:
        post_processed_errs=post_process(errs, confidence_level=confidence_level)

    if post_processed_diff==None and errs!=None:
        post_processed_diff=post_process((errs[0]-errs[1]).unsqueeze(0), confidence_level=confidence_level)[0]
    
    rows=1

    cols=len(confidence_level)

    if post_processed_diff!=None:
        rows+=1

    if post_processed_errs[0]["accuracy"].shape[-1]==2:
        rows+=1

    fig, axs =plt.subplots(rows, cols, figsize=(6*cols,6*rows))
    

    xs=range(len(post_processed_errs[0]["accuracy"][:, 0]))
    for name, results in zip(names, post_processed_errs): #plots the pure losses
        for i, level in enumerate(confidence_level):
            axs[0, i].plot(results["accuracy"][:, 0], label=name)
            axs[0, i].fill_between(xs, results[f"confidence_{level}_lower"][:, 0], results[f"confidence_{level}_upper"][:, 0], alpha=0.5)
            axs[0, i].set_title(f'Pure loss, confidence {level}')
            axs[0, i].set_ylim(bottom=ylim[0], top=ylim[1])


    if post_processed_errs[0]["accuracy"].shape[-1]==2: #plots losses relative to optimal predictor
        for name, results in zip(names, post_processed_errs):
            for i, level in enumerate(confidence_level):
                axs[1, i].plot(results["accuracy"][:, 1], label=name)
                axs[1, i].fill_between(xs, results[f"confidence_{level}_lower"][:, 1], results[f"confidence_{level}_upper"][:, 1], alpha=0.5)
                axs[1, i].set_title(f'Pure loss, confidence {level}')

    if post_processed_diff!=None: #plots difference between losses
        for i, level in enumerate(confidence_level):
            axs[-1, i].plot(post_processed_diff["accuracy"][:, 0])
            axs[-1, i].fill_between(xs, post_processed_diff[f"confidence_{level}_lower"][:, 0], post_processed_diff[f"confidence_{level}_upper"][:, 0], alpha=0.5)
            axs[-1, i].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
            axs[-1, i].set_title(f'Loss of {names[0]} minus loss of {names[1]}, confidence {level}')
    if title!=None:
        fig.suptitle(title)
    lines, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(lines, labels, bbox_to_anchor=(0.5, 0.95) , loc='upper center', ncol=len(MODELS))
    plt.show()

