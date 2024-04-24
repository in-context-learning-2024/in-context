import torch

from typing import Iterable
from torch import Tensor

from scipy.stats import norm

def post_process(results: Iterable[Tensor],
                 confidence_level: list[float] = [0.01, 0.05],
                 quantile_cutoffs: list[float] = [0.05, 0.25, 0.5, 0.75, 0.95],
                ) -> list[dict[str, Tensor]]:
    summaries: list[dict[str, Tensor]] = [ ]
    for err_tensor in results:
	
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
            interval_jump = norm.ppf(1 - level/2) * std_err_of_mean
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
