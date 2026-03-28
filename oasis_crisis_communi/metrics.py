"""
Metrics and output generation for the OASIS Crisis Communication Optimizer.

Provides functions to:
- Convert SimulationResult objects to flat rows
- Save belief alignment data to CSV
- Generate comparison plots
- Compute statistical summaries
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .simulation import SimulationResult

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np

# numpy 2.0 renamed trapz → trapezoid
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz  # type: ignore[attr-defined]

# matplotlib 3.7+ deprecates cm.get_cmap; use colormaps dict directly
try:
    _get_cmap = matplotlib.colormaps.__getitem__
except AttributeError:
    import matplotlib.cm as _cm  # type: ignore[assignment]
    _get_cmap = _cm.get_cmap  # type: ignore[assignment]


OUTPUTS_DIR = Path(os.getenv("OUTPUT_DIR", str(Path(__file__).parent.parent / "outputs")))


def results_to_rows(results: "List[SimulationResult]") -> List[Dict]:
    """
    Convert a list of SimulationResult objects to flat rows for CSV export.

    Each row has keys: strategy, timestep, alignment_score
    """
    rows = []
    for result in results:
        for t, score in enumerate(result.belief_timeline):
            rows.append(
                {
                    "strategy": result.strategy.name,
                    "timestep": t,
                    "alignment_score": round(float(score), 4),
                }
            )
    return rows


def save_alignment_csv(
    rows: List[Dict],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save belief alignment data to CSV.

    Parameters
    ----------
    rows : list of dicts
        Each dict must have keys: strategy, timestep, alignment_score
    output_path : Path, optional
        Defaults to outputs/belief_alignment.csv.
    """
    if output_path is None:
        output_path = OUTPUTS_DIR / "belief_alignment.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows, columns=["strategy", "timestep", "alignment_score"])
    df.to_csv(output_path, index=False)
    return output_path


def load_alignment_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """Load belief alignment CSV."""
    if path is None:
        path = OUTPUTS_DIR / "belief_alignment.csv"
    return pd.read_csv(path)


def plot_strategy_comparison(
    rows: List[Dict],
    output_path: Optional[Path] = None,
    title: str = "Crisis Communication Strategy Comparison",
) -> Path:
    """
    Generate a line plot comparing alignment over time for each strategy.

    Parameters
    ----------
    rows : list of dicts with keys: strategy, timestep, alignment_score
    output_path : Path, optional
        Defaults to outputs/strategy_comparison.png.
    """
    if output_path is None:
        output_path = OUTPUTS_DIR / "strategy_comparison.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows, columns=["strategy", "timestep", "alignment_score"])
    strategies = df["strategy"].unique()
    cmap = _get_cmap("tab10")
    n_strats = len(strategies)

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, strat_name in enumerate(strategies):
        subset = df[df["strategy"] == strat_name].sort_values("timestep")
        color = cmap(idx / max(1, n_strats - 1)) if n_strats > 1 else cmap(0.5)
        ax.plot(
            subset["timestep"],
            subset["alignment_score"],
            label=strat_name,
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=4,
        )

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="50% baseline")
    ax.set_xlabel("Simulation Timestep", fontsize=12)
    ax.set_ylabel("Belief Alignment Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")
    fig.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compute_summary_table(rows: List[Dict]) -> pd.DataFrame:
    """
    Compute summary statistics per strategy.

    Returns a DataFrame with columns:
    strategy, final_score, peak_score, mean_score, area_under_curve
    """
    df = pd.DataFrame(rows, columns=["strategy", "timestep", "alignment_score"])
    records = []
    for strat in df["strategy"].unique():
        sub = df[df["strategy"] == strat].sort_values("timestep")
        scores = sub["alignment_score"].values
        records.append(
            {
                "strategy": strat,
                "final_score": round(float(scores[-1]), 4),
                "peak_score": round(float(scores.max()), 4),
                "mean_score": round(float(scores.mean()), 4),
                "area_under_curve": round(float(_trapz(scores) / len(scores)), 4),
            }
        )
    return pd.DataFrame(records).sort_values("final_score", ascending=False)


def identify_winner(rows: List[Dict]) -> str:
    """Return the strategy name with the highest final alignment score."""
    summary = compute_summary_table(rows)
    return str(summary.iloc[0]["strategy"])
