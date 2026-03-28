#!/usr/bin/env python3
"""
Demo script for the OASIS Crisis Communication Optimizer.

Runs a 3-strategy simulation and generates:
- outputs/belief_alignment.csv
- outputs/strategy_comparison.png

Usage:
    python scripts/demo.py
"""

import sys
import os

# Allow running from repo root or scripts/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

# Optional rich output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    _RICH = True
    console = Console()
except ImportError:
    _RICH = False
    class _FallbackConsole:
        def print(self, *a, **kw): print(*a)
        def rule(self, *a, **kw): print("─" * 60)
    console = _FallbackConsole()

from oasis_crisis_communi.strategies import CommunicationStrategy
from oasis_crisis_communi.simulation import CrisisSimulation
from oasis_crisis_communi.metrics import (
    results_to_rows,
    save_alignment_csv,
    plot_strategy_comparison,
    compute_summary_table,
    identify_winner,
)
from oasis_crisis_communi.belief_tracker import BeliefTracker


def main():
    console.rule("[bold blue]OASIS Crisis Communication Optimizer — Demo" if _RICH
                 else "OASIS Crisis Communication Optimizer — Demo")

    strategies = CommunicationStrategy.all_defaults()
    console.print(
        f"\n[bold]Running {len(strategies)}-strategy simulation...[/bold]" if _RICH
        else f"\nRunning {len(strategies)}-strategy simulation..."
    )
    console.print(f"Strategies: {[s.name for s in strategies]}")

    # Run simulation
    sim = CrisisSimulation(num_agents=50, num_timesteps=20, num_misinfo_agents=5, seed=42)
    results = sim.run_comparison(strategies)

    # Convert to flat rows for CSV/plot
    rows = results_to_rows(results)

    # Save outputs
    csv_path = save_alignment_csv(rows)
    plot_path = plot_strategy_comparison(rows)

    console.print(f"\n[green]CSV saved:[/green] {csv_path}" if _RICH else f"\nCSV saved: {csv_path}")
    console.print(f"[green]Plot saved:[/green] {plot_path}" if _RICH else f"Plot saved: {plot_path}")

    # Summary table
    summary = compute_summary_table(rows)
    winner = identify_winner(rows)

    if _RICH:
        table = Table(
            title="Strategy Comparison Summary",
            show_header=True,
            header_style="bold cyan",
        )
        for col in summary.columns:
            table.add_column(col)
        for _, row in summary.iterrows():
            table.add_row(*[str(v) for v in row])
        console.print(table)
    else:
        print("\n" + summary.to_string(index=False))

    console.print(
        f"\n[bold green]Winning strategy: {winner}[/bold green]" if _RICH
        else f"\nWinning strategy: {winner}"
    )

    winning_score = summary[summary["strategy"] == winner]["final_score"].values[0]
    if winning_score > 0.5:
        console.print(
            f"[green]Final alignment score {winning_score:.3f} > 0.5 threshold — success![/green]"
            if _RICH
            else f"Final alignment score {winning_score:.3f} > 0.5 threshold — success!"
        )
    else:
        console.print(
            f"[yellow]Final alignment score {winning_score:.3f} (below 0.5)[/yellow]"
            if _RICH
            else f"Final alignment score {winning_score:.3f} (below 0.5)"
        )

    # Demo INTERVIEW action via BeliefTracker
    tracker = BeliefTracker()
    sample_response = tracker.parse_interview_response(
        agent_id=99,
        timestep=10,
        raw_response=(
            "I trust the official guidance. The information seems credible "
            "and based on solid evidence. I believe we should follow the recommendations."
        ),
    )

    console.print(
        f"\n[bold]Sample INTERVIEW response:[/bold]" if _RICH
        else "\nSample INTERVIEW response:"
    )
    if _RICH:
        console.print(Panel(
            f"Agent ID: {sample_response.agent_id}\n"
            f"Timestep: {sample_response.timestep}\n"
            f"Belief Score: {sample_response.belief_score}\n"
            f"Sentiment: {sample_response.sentiment}\n"
            f"Keywords: {sample_response.keywords[:5]}",
            title="INTERVIEW Action",
        ))
    else:
        for k, v in sample_response.to_dict().items():
            print(f"  {k}: {v}")

    console.rule("[bold blue]Demo Complete" if _RICH else "Demo Complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
