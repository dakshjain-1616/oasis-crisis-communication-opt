"""
Sensitivity analysis for crisis communication simulations.

Runs a strategy across multiple random seeds to compute mean belief
timelines with 95% confidence intervals, quantifying how robust a
strategy's performance is across different population conditions.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from .strategies import StrategyConfig
from .simulation import MockSimulation


@dataclass
class SensitivityResult:
    """Aggregated results from a multi-seed sensitivity run."""
    strategy_name: str
    n_seeds: int
    seeds: List[int]
    mean_belief_timeline: List[float]    # mean alignment at each timestep
    ci_lower_95: List[float]             # 2.5th percentile per timestep
    ci_upper_95: List[float]             # 97.5th percentile per timestep
    std_timeline: List[float]            # std dev per timestep
    final_alignment_mean: float
    final_alignment_std: float
    final_alignment_min: float = 0.0
    final_alignment_max: float = 0.0

    def summary(self) -> dict:
        """Return a concise summary dict."""
        return {
            "strategy": self.strategy_name,
            "n_seeds": self.n_seeds,
            "final_mean": round(self.final_alignment_mean, 4),
            "final_std": round(self.final_alignment_std, 4),
            "final_min": round(self.final_alignment_min, 4),
            "final_max": round(self.final_alignment_max, 4),
            "ci_width_final": round(
                self.ci_upper_95[-1] - self.ci_lower_95[-1], 4
            ) if self.ci_upper_95 else 0.0,
        }


def run_sensitivity(
    strategy: StrategyConfig,
    n_seeds: int = 20,
    base_seed: int = 0,
    num_agents: int = 30,
    num_timesteps: int = 20,
    num_misinfo_agents: int = 5,
) -> SensitivityResult:
    """
    Run a strategy across ``n_seeds`` random seeds and aggregate results.

    Parameters
    ----------
    strategy : StrategyConfig
        The communication strategy to analyse.
    n_seeds : int
        Number of independent seeds to run (more = tighter CI).
    base_seed : int
        First seed; subsequent seeds are base_seed+1, base_seed+2, ...
    num_agents : int
        Number of public agents per run.
    num_timesteps : int
        Simulation length per run.
    num_misinfo_agents : int
        Misinformation agents per run.

    Returns
    -------
    SensitivityResult
    """
    if n_seeds < 2:
        raise ValueError(f"n_seeds must be >= 2 for CI computation, got {n_seeds}")

    seeds = list(range(base_seed, base_seed + n_seeds))
    timelines: List[List[float]] = []

    for seed in seeds:
        sim = MockSimulation(
            strategy=strategy,
            num_agents=num_agents,
            num_timesteps=num_timesteps,
            num_misinfo_agents=num_misinfo_agents,
            seed=seed,
        )
        result = sim.run()
        timelines.append(result.belief_timeline)

    arr = np.array(timelines, dtype=float)   # shape: (n_seeds, n_timesteps)
    mean_tl = arr.mean(axis=0).tolist()
    lower = np.percentile(arr, 2.5, axis=0).tolist()
    upper = np.percentile(arr, 97.5, axis=0).tolist()
    std_tl = arr.std(axis=0).tolist()
    finals = arr[:, -1]

    return SensitivityResult(
        strategy_name=strategy.name,
        n_seeds=n_seeds,
        seeds=seeds,
        mean_belief_timeline=[round(v, 4) for v in mean_tl],
        ci_lower_95=[round(v, 4) for v in lower],
        ci_upper_95=[round(v, 4) for v in upper],
        std_timeline=[round(v, 4) for v in std_tl],
        final_alignment_mean=round(float(finals.mean()), 4),
        final_alignment_std=round(float(finals.std()), 4),
        final_alignment_min=round(float(finals.min()), 4),
        final_alignment_max=round(float(finals.max()), 4),
    )


def compare_sensitivity(
    strategies: List[StrategyConfig],
    n_seeds: int = 20,
    base_seed: int = 0,
    num_agents: int = 30,
    num_timesteps: int = 20,
    num_misinfo_agents: int = 5,
) -> List[SensitivityResult]:
    """
    Run sensitivity analysis for multiple strategies.

    Returns one SensitivityResult per strategy in the same order.
    """
    return [
        run_sensitivity(
            strategy=s,
            n_seeds=n_seeds,
            base_seed=base_seed,
            num_agents=num_agents,
            num_timesteps=num_timesteps,
            num_misinfo_agents=num_misinfo_agents,
        )
        for s in strategies
    ]
