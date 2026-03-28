"""Results analysis and strategy comparison."""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd
try:
    from scipy import stats as _stats  # optional; not required for core analysis
except ImportError:
    _stats = None  # type: ignore[assignment]

from .simulation import SimulationResult
from .strategies import StrategyConfig


class ResultsAnalyzer:
    """Analyze simulation results and generate comparative insights."""

    def __init__(self, results: List[SimulationResult]):
        self.results = results

    def to_dataframe(self) -> pd.DataFrame:
        """Convert belief timelines to a tidy DataFrame."""
        rows = []
        for result in self.results:
            for t, score in enumerate(result.belief_timeline, start=1):
                rows.append({
                    "strategy": result.strategy.name,
                    "timestep": t,
                    "alignment_score": round(score, 4),
                    "repost_rate": round(result.repost_rates[t - 1], 4),
                    "misinfo_repost_rate": round(result.misinfo_repost_rates[t - 1], 4),
                })
        return pd.DataFrame(rows)

    def get_winner(self) -> Optional[SimulationResult]:
        """Return the winning strategy result."""
        winning = [r for r in self.results if r.winning]
        if winning:
            return winning[0]
        if self.results:
            return max(self.results, key=lambda r: r.final_alignment)
        return None

    def get_summary_table(self) -> pd.DataFrame:
        """Generate a summary table of all strategies."""
        rows = [r.to_summary_dict() for r in self.results]
        df = pd.DataFrame(rows)
        df = df.sort_values("final_alignment", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df

    def compute_effect_sizes(self) -> dict:
        """Compute Cohen's d effect sizes between strategy pairs."""
        effect_sizes = {}
        for i, r1 in enumerate(self.results):
            for j, r2 in enumerate(self.results):
                if i >= j:
                    continue
                a1 = np.array(r1.belief_timeline)
                a2 = np.array(r2.belief_timeline)
                pooled_std = np.sqrt((np.std(a1) ** 2 + np.std(a2) ** 2) / 2)
                if pooled_std > 0:
                    d = (np.mean(a1) - np.mean(a2)) / pooled_std
                else:
                    d = 0.0
                key = f"{r1.strategy.name} vs {r2.strategy.name}"
                effect_sizes[key] = round(float(d), 3)
        return effect_sizes

    def compute_time_to_threshold(self, threshold: float = 0.6) -> dict:
        """Find timestep at which each strategy first exceeds threshold alignment."""
        result = {}
        for r in self.results:
            exceeded = [t + 1 for t, score in enumerate(r.belief_timeline) if score >= threshold]
            result[r.strategy.name] = exceeded[0] if exceeded else None
        return result

    def generate_recommendation(self) -> str:
        """Generate a text recommendation based on analysis."""
        winner = self.get_winner()
        if not winner:
            return "Insufficient data to generate recommendation."

        effect_sizes = self.compute_effect_sizes()
        threshold_times = self.compute_time_to_threshold(0.6)

        lines = [
            f"WINNING STRATEGY: {winner.strategy.name}",
            f"Final Alignment Score: {winner.final_alignment:.1%}",
            "",
            f"Description: {winner.strategy.description}",
            "",
            "Key Findings:",
        ]

        # Add comparative insights
        sorted_results = sorted(self.results, key=lambda r: r.final_alignment, reverse=True)
        for rank, r in enumerate(sorted_results, 1):
            tt = threshold_times.get(r.strategy.name)
            tt_str = f"timestep {tt}" if tt else "never reached 60%"
            lines.append(
                f"  {rank}. {r.strategy.name}: "
                f"{r.final_alignment:.1%} final alignment, "
                f"60% threshold at {tt_str}"
            )

        lines += [
            "",
            "Recommendations for Public Health Communicators:",
            f"  • Start communications early (by timestep {winner.strategy.start_timestep})",
            f"  • Use {winner.strategy.tone_label} tone in messaging",
            f"  • Post {winner.strategy.posts_per_timestep}x per communication cycle",
        ]

        if winner.strategy.multi_channel:
            lines.append("  • Deploy across multiple channels for maximum reach")

        if effect_sizes:
            best_effect = max(effect_sizes.items(), key=lambda x: abs(x[1]))
            lines.append(f"  • Largest effect size: {best_effect[0]} (d={best_effect[1]:.2f})")

        return "\n".join(lines)

    def compute_influencer_impact(self) -> dict:
        """
        Return per-strategy influencer and belief-distribution statistics.

        Uses ``belief_distribution`` and ``influencer_fraction`` from
        ``SimulationResult`` (populated when running MockSimulation ≥ v2).
        Returns an empty dict for older results that lack these fields.
        """
        impact = {}
        for r in self.results:
            if not r.belief_distribution:
                continue
            dist = np.array(r.belief_distribution, dtype=float)
            impact[r.strategy.name] = {
                "influencer_fraction": round(float(r.influencer_fraction), 4),
                "belief_mean": round(float(dist.mean()), 4),
                "belief_median": round(float(np.median(dist)), 4),
                "belief_std": round(float(dist.std()), 4),
                "pct_above_60": round(float((dist >= 0.6).mean()), 4),
                "pct_below_40": round(float((dist < 0.4).mean()), 4),
            }
        return impact

    def get_belief_histogram_data(
        self, result: "SimulationResult", n_bins: int = 10
    ) -> dict:
        """
        Return histogram bin data for the final agent belief distribution.

        Parameters
        ----------
        result : SimulationResult
        n_bins : int, default 10

        Returns
        -------
        dict with keys: bin_edges, counts, bin_centers
        """
        if not result.belief_distribution:
            return {"bin_edges": [], "counts": [], "bin_centers": []}
        dist = np.array(result.belief_distribution, dtype=float)
        counts, edges = np.histogram(dist, bins=n_bins, range=(0.0, 1.0))
        centers = [round((edges[i] + edges[i + 1]) / 2, 3) for i in range(len(counts))]
        return {
            "bin_edges": edges.tolist(),
            "counts": counts.tolist(),
            "bin_centers": centers,
        }

    def get_run_stats(self) -> dict:
        """Return timing and token stats aggregated across all strategies."""
        return {
            r.strategy.name: {
                "run_timestamp": r.run_timestamp,
                "run_duration_sec": r.run_duration_sec,
                "token_estimate": r.token_estimate,
            }
            for r in self.results
        }

    def get_interview_sample(self, n: int = 5) -> List[dict]:
        """Get a sample of interview responses from the winning strategy."""
        winner = self.get_winner()
        if not winner:
            return []
        responses = winner.interview_responses
        if not responses:
            return []
        # Sample from last few timesteps
        last_ts = max(r["timestep"] for r in responses)
        late_responses = [r for r in responses if r["timestep"] >= last_ts - 2]
        import random
        rng = random.Random(42)
        return rng.sample(late_responses, min(n, len(late_responses)))
