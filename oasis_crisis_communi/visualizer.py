"""Matplotlib visualization for simulation results."""
from __future__ import annotations

import os
from typing import List, Optional, TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .simulation import SimulationResult
from .analyzer import ResultsAnalyzer

if TYPE_CHECKING:
    from .sensitivity import SensitivityResult


class SimulationVisualizer:
    """Generate publication-quality visualizations of simulation results."""

    DEFAULT_STYLE = {
        "figure.facecolor": "#FAFAFA",
        "axes.facecolor": "#FFFFFF",
        "axes.grid": True,
        "grid.alpha": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
    }

    def __init__(self, results: List[SimulationResult], output_dir: str = "outputs"):
        self.results = results
        self.analyzer = ResultsAnalyzer(results)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_belief_alignment(self, save_path: Optional[str] = None) -> str:
        """Plot belief alignment curves for all strategies."""
        save_path = save_path or os.path.join(self.output_dir, "strategy_comparison.png")

        with plt.rc_context(self.DEFAULT_STYLE):
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(
                "OASIS Crisis Communication Strategy Comparison\nBelief Alignment Over Time",
                fontsize=14, fontweight="bold", y=0.98
            )

            # Main comparison plot (top left, spanning full width if <=3 strategies)
            ax_main = axes[0, 0]
            ax_repost = axes[0, 1]
            ax_misinfo = axes[1, 0]
            ax_summary = axes[1, 1]

            timesteps = list(range(1, len(self.results[0].belief_timeline) + 1))

            # --- Main: Belief Alignment ---
            ax_main.set_title("Belief Alignment Over Time", fontweight="bold")
            for result in self.results:
                color = result.strategy.color
                label = result.strategy.name
                if result.winning:
                    label += " \u2605"
                    lw = 3.0
                    ls = "-"
                else:
                    lw = 1.8
                    ls = "--"

                smoothed = self._smooth(result.belief_timeline)
                ax_main.plot(timesteps, smoothed, color=color, lw=lw, ls=ls, label=label)
                # Confidence band (mock uncertainty)
                std = np.std(result.belief_timeline) * 0.3
                ax_main.fill_between(
                    timesteps,
                    [max(0, v - std) for v in smoothed],
                    [min(1, v + std) for v in smoothed],
                    color=color, alpha=0.12
                )

            ax_main.axhline(0.6, color="gray", lw=1, ls=":", alpha=0.7, label="60% threshold")
            ax_main.set_xlabel("Timestep")
            ax_main.set_ylabel("Mean Belief Alignment (0-1)")
            ax_main.set_ylim(0, 1)
            ax_main.legend(loc="lower right", fontsize=8)

            # --- Repost Rates ---
            ax_repost.set_title("Government Post Repost Rate", fontweight="bold")
            for result in self.results:
                ax_repost.plot(
                    timesteps, result.repost_rates,
                    color=result.strategy.color, lw=1.8,
                    label=result.strategy.name,
                )
            ax_repost.set_xlabel("Timestep")
            ax_repost.set_ylabel("Repost Rate")
            ax_repost.set_ylim(0, 1)
            ax_repost.legend(loc="upper left", fontsize=8)

            # --- Misinfo Repost Rates ---
            ax_misinfo.set_title("Misinformation Repost Rate", fontweight="bold")
            for result in self.results:
                ax_misinfo.plot(
                    timesteps, result.misinfo_repost_rates,
                    color=result.strategy.color, lw=1.8, ls="--",
                    label=result.strategy.name,
                )
            ax_misinfo.set_xlabel("Timestep")
            ax_misinfo.set_ylabel("Misinfo Repost Rate")
            ax_misinfo.set_ylim(0, 1)
            ax_misinfo.legend(loc="upper right", fontsize=8)

            # --- Summary bar chart ---
            ax_summary.set_title("Final Alignment Score by Strategy", fontweight="bold")
            names = [r.strategy.name for r in self.results]
            scores = [r.final_alignment for r in self.results]
            colors = [r.strategy.color for r in self.results]
            bars = ax_summary.bar(range(len(names)), scores, color=colors, alpha=0.85, edgecolor="white")
            ax_summary.set_xticks(range(len(names)))
            ax_summary.set_xticklabels(
                [n.replace(" & ", "\n& ") for n in names],
                fontsize=8
            )
            ax_summary.set_ylabel("Final Alignment Score")
            ax_summary.set_ylim(0, 1)
            for bar, score in zip(bars, scores):
                ax_summary.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{score:.1%}",
                    ha="center", fontsize=9, fontweight="bold"
                )

            plt.tight_layout()
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return save_path

    def plot_single_strategy(self, result: SimulationResult, save_path: Optional[str] = None) -> str:
        """Detailed plot for a single strategy."""
        name_slug = result.strategy.name.lower().replace(" ", "_").replace("&", "and")
        save_path = save_path or os.path.join(self.output_dir, f"strategy_{name_slug}.png")

        timesteps = list(range(1, len(result.belief_timeline) + 1))
        color = result.strategy.color

        with plt.rc_context(self.DEFAULT_STYLE):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            fig.suptitle(
                f"Strategy: {result.strategy.name}\n{result.strategy.description}",
                fontsize=12, fontweight="bold"
            )

            ax1.plot(timesteps, result.belief_timeline, color=color, lw=2)
            ax1.fill_between(timesteps, 0, result.belief_timeline, color=color, alpha=0.15)
            ax1.axhline(0.6, color="gray", lw=1, ls=":", label="60% threshold")
            if result.strategy.start_timestep > 1:
                ax1.axvline(
                    result.strategy.start_timestep,
                    color="orange", lw=1.5, ls="--",
                    label=f"Gov starts at t={result.strategy.start_timestep}"
                )
            ax1.set_ylabel("Mean Belief Alignment")
            ax1.set_ylim(0, 1)
            ax1.legend()

            ax2.plot(timesteps, result.repost_rates, color=color, lw=1.8, label="Gov reposts")
            ax2.plot(timesteps, result.misinfo_repost_rates, color="#E53935", lw=1.8, ls="--", label="Misinfo reposts")
            ax2.set_xlabel("Timestep")
            ax2.set_ylabel("Repost Rate")
            ax2.set_ylim(0, 1)
            ax2.legend()

            plt.tight_layout()
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return save_path

    def plot_belief_distribution(
        self,
        result: SimulationResult,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Plot a histogram of the final agent belief distribution.

        Shows how belief is spread across the population at simulation end,
        including mean, median, and the 60% alignment threshold.
        """
        name_slug = result.strategy.name.lower().replace(" ", "_").replace("&", "and")
        save_path = save_path or os.path.join(
            self.output_dir, f"belief_dist_{name_slug}.png"
        )

        with plt.rc_context(self.DEFAULT_STYLE):
            fig, ax = plt.subplots(figsize=(8, 5))
            color = result.strategy.color

            if result.belief_distribution:
                dist = np.array(result.belief_distribution)
                ax.hist(dist, bins=20, range=(0.0, 1.0), color=color, alpha=0.75,
                        edgecolor="white", linewidth=0.8)
                mean_val = float(dist.mean())
                median_val = float(np.median(dist))
                ax.axvline(mean_val, color=color, lw=2, ls="-",
                           label=f"Mean: {mean_val:.2f}")
                ax.axvline(median_val, color=color, lw=1.5, ls="--",
                           label=f"Median: {median_val:.2f}")
            else:
                ax.text(0.5, 0.5, "No distribution data available",
                        ha="center", va="center", transform=ax.transAxes, fontsize=11)

            ax.axvline(0.6, color="gray", lw=1.2, ls=":", alpha=0.8,
                       label="60% threshold")
            ax.set_xlabel("Final Belief Score (0=distrust, 1=full trust)", fontsize=11)
            ax.set_ylabel("Number of Agents", fontsize=11)
            ax.set_title(
                f"Final Belief Distribution — {result.strategy.name}",
                fontsize=13, fontweight="bold",
            )
            ax.set_xlim(0, 1)
            ax.legend(fontsize=9)

            plt.tight_layout()
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return save_path

    def plot_sensitivity_bands(
        self,
        sensitivity_results: "List[SensitivityResult]",
        save_path: Optional[str] = None,
    ) -> str:
        """
        Plot mean belief timelines with 95% CI shading for each strategy.

        Requires sensitivity_results from sensitivity.compare_sensitivity().
        """
        save_path = save_path or os.path.join(self.output_dir, "sensitivity_bands.png")

        with plt.rc_context(self.DEFAULT_STYLE):
            fig, ax = plt.subplots(figsize=(10, 6))

            colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800",
                      "#9C27B0", "#00BCD4", "#FF5722"]

            for idx, sr in enumerate(sensitivity_results):
                color = colors[idx % len(colors)]
                timesteps = list(range(1, len(sr.mean_belief_timeline) + 1))
                ax.plot(timesteps, sr.mean_belief_timeline, color=color, lw=2.5,
                        label=f"{sr.strategy_name} (mean)")
                ax.fill_between(
                    timesteps, sr.ci_lower_95, sr.ci_upper_95,
                    color=color, alpha=0.18,
                    label=f"{sr.strategy_name} 95% CI",
                )

            ax.axhline(0.6, color="gray", lw=1, ls=":", alpha=0.7, label="60% threshold")
            ax.set_xlabel("Simulation Timestep", fontsize=12)
            ax.set_ylabel("Mean Belief Alignment", fontsize=12)
            ax.set_title(
                f"Strategy Robustness — {sensitivity_results[0].n_seeds} Seeds, 95% CI",
                fontsize=13, fontweight="bold",
            )
            ax.set_ylim(0, 1)
            # Deduplicate legend labels (mean + CI band = 2 entries per strategy)
            handles, labels = ax.get_legend_handles_labels()
            seen: dict = {}
            for h, l in zip(handles, labels):
                base = l.split(" (")[0]
                if base not in seen:
                    seen[base] = h
            ax.legend(seen.values(), seen.keys(), loc="lower right", fontsize=9)

            plt.tight_layout()
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return save_path

    @staticmethod
    def _smooth(values: list, window: int = 3) -> list:
        """Apply a simple moving average for smoother curves."""
        if len(values) < window:
            return values
        result = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            result.append(sum(values[start:end]) / (end - start))
        return result
