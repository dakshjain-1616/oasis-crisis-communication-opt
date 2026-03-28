"""
OASIS Crisis Communication Optimizer — Demo Script

Runs a 3-strategy comparison with 30 agents over 20 timesteps.
Works in mock mode (no API key needed) and real OASIS/LLM mode.

Usage:
    python demo.py
"""
from __future__ import annotations

import os
import sys
import logging
import json
import datetime

# Load .env if present
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Rich console for pretty output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich import print as rprint
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

import pandas as pd

from oasis_crisis_communi.strategies import CommunicationStrategy
from oasis_crisis_communi.simulation import CrisisSimulation, MOCK_MODE
from oasis_crisis_communi.analyzer import ResultsAnalyzer
from oasis_crisis_communi.visualizer import SimulationVisualizer
from oasis_crisis_communi.reporter import ReportGenerator


OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
NUM_AGENTS = int(os.getenv("NUM_AGENTS", "30"))
NUM_TIMESTEPS = int(os.getenv("NUM_TIMESTEPS", "20"))
NUM_MISINFO_AGENTS = int(os.getenv("NUM_MISINFO_AGENTS", "5"))
SEED = int(os.getenv("RANDOM_SEED", "42"))


def print_header():
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold blue]OASIS Crisis Communication Optimizer[/bold blue]\n"
            "[dim]Simulating public health emergency communication strategies[/dim]",
            border_style="blue"
        ))
        mode_str = "[yellow]MOCK MODE[/yellow] (no API key)" if MOCK_MODE else "[green]LIVE MODE[/green] (LLM agents)"
        console.print(f"  Mode: {mode_str}")
        console.print(f"  Agents: [cyan]{NUM_AGENTS}[/cyan]  |  "
                      f"Timesteps: [cyan]{NUM_TIMESTEPS}[/cyan]  |  "
                      f"Misinfo agents: [cyan]{NUM_MISINFO_AGENTS}[/cyan]")
        console.print()
    else:
        print("=" * 60)
        print("OASIS Crisis Communication Optimizer")
        print("=" * 60)
        mode_str = "MOCK MODE (no API key)" if MOCK_MODE else "LIVE MODE (LLM agents)"
        print(f"Mode: {mode_str}")
        print(f"Agents: {NUM_AGENTS} | Timesteps: {NUM_TIMESTEPS} | Misinfo: {NUM_MISINFO_AGENTS}")
        print()


def print_results(results, analyzer):
    if RICH_AVAILABLE:
        summary_df = analyzer.get_summary_table()
        table = Table(title="Strategy Comparison Results", show_header=True, header_style="bold blue")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Strategy", min_width=25)
        table.add_column("Final Alignment", justify="right")
        table.add_column("Peak", justify="right")
        table.add_column("Avg", justify="right")
        table.add_column("Gov Reposts", justify="right")

        for _, row in summary_df.iterrows():
            winner_badge = " 🏆" if row.get("winning") else ""
            style = "bold green" if row.get("winning") else ""
            table.add_row(
                str(int(row["rank"])),
                f"{row['strategy']}{winner_badge}",
                f"{float(row['final_alignment']):.1%}",
                f"{float(row['peak_alignment']):.1%}",
                f"{float(row['avg_alignment']):.1%}",
                str(int(row["total_gov_reposts"])),
                style=style,
            )
        console.print(table)
        console.print()
    else:
        print("\nStrategy Comparison Results:")
        print("-" * 70)
        summary_df = analyzer.get_summary_table()
        print(summary_df.to_string(index=False))
        print()


def print_recommendation(recommendation: str):
    if RICH_AVAILABLE:
        console.print(Panel(
            recommendation,
            title="[bold green]Recommendation[/bold green]",
            border_style="green"
        ))
    else:
        print("\n" + "=" * 60)
        print("RECOMMENDATION")
        print("=" * 60)
        print(recommendation)
        print()


def run_demo():
    """Main demo execution."""
    print_header()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define the three strategies
    strategies = CommunicationStrategy.all_defaults()

    if RICH_AVAILABLE:
        console.print("[bold]Strategies to compare:[/bold]")
        for s in strategies:
            console.print(
                f"  • [cyan]{s.name}[/cyan]: start_t={s.start_timestep}, "
                f"tone={s.tone_label}, freq={s.posts_per_timestep}x, "
                f"multi_channel={s.multi_channel}"
            )
        console.print()
    else:
        print("Strategies to compare:")
        for s in strategies:
            print(f"  - {s.name}: start_t={s.start_timestep}, tone={s.tone_label}, "
                  f"freq={s.posts_per_timestep}x, multi_channel={s.multi_channel}")
        print()

    # Run simulation
    sim = CrisisSimulation(
        num_agents=NUM_AGENTS,
        num_timesteps=NUM_TIMESTEPS,
        num_misinfo_agents=NUM_MISINFO_AGENTS,
        seed=SEED,
    )

    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running simulations...", total=len(strategies))
            results = []
            for strategy in strategies:
                progress.update(task, description=f"Running: {strategy.name}")
                result = sim.run_strategy(strategy)
                results.append(result)
                progress.advance(task)
        # Mark winner
        best = max(results, key=lambda r: r.final_alignment)
        best.winning = True
    else:
        print("Running simulations...")
        results = sim.run_comparison(strategies)

    # Analyze
    analyzer = ResultsAnalyzer(results)
    print_results(results, analyzer)

    # Generate recommendation
    recommendation = analyzer.generate_recommendation()
    print_recommendation(recommendation)

    # Save CSV
    df = analyzer.to_dataframe()
    csv_path = os.path.join(OUTPUT_DIR, "belief_alignment.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path}")

    # Save visualization
    if RICH_AVAILABLE:
        console.print("[dim]Generating visualizations...[/dim]")
    else:
        print("Generating visualizations...")

    viz = SimulationVisualizer(results, output_dir=OUTPUT_DIR)
    chart_path = viz.plot_belief_alignment()
    logger.info(f"Saved chart: {chart_path}")

    # Save individual strategy charts
    for result in results:
        single_path = viz.plot_single_strategy(result)
        logger.info(f"Saved strategy chart: {single_path}")

    # Save reports
    reporter = ReportGenerator(results, output_dir=OUTPUT_DIR)
    json_path = reporter.save_json_report()
    logger.info(f"Saved JSON report: {json_path}")

    html_path = reporter.save_html_report()
    logger.info(f"Saved HTML report: {html_path}")

    # Save interview sample
    interview_path = os.path.join(OUTPUT_DIR, "interview_samples.json")
    samples = analyzer.get_interview_sample(10)
    with open(interview_path, "w") as f:
        json.dump(samples, f, indent=2)
    logger.info(f"Saved interview samples: {interview_path}")

    # Print summary of output files
    winner = analyzer.get_winner()
    if RICH_AVAILABLE:
        console.print()
        console.print(Panel.fit(
            f"[bold green]✓ Demo complete![/bold green]\n\n"
            f"[bold]Output files:[/bold]\n"
            f"  📊 {csv_path}\n"
            f"  📈 {chart_path}\n"
            f"  📄 {html_path}\n"
            f"  🔬 {json_path}\n"
            f"  💬 {interview_path}\n\n"
            f"[bold]Winner:[/bold] {winner.strategy.name if winner else 'N/A'} "
            f"({winner.final_alignment:.1%} alignment)" if winner else "",
            border_style="green"
        ))
    else:
        print("\nDemo complete!")
        print(f"Output files:")
        print(f"  {csv_path}")
        print(f"  {chart_path}")
        print(f"  {html_path}")
        print(f"  {json_path}")
        print(f"  {interview_path}")
        if winner:
            print(f"\nWinner: {winner.strategy.name} ({winner.final_alignment:.1%} alignment)")

    return results, analyzer


if __name__ == "__main__":
    run_demo()
