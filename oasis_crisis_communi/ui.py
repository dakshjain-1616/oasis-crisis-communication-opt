"""
Gradio UI for the OASIS Crisis Communication Optimizer.

Provides an interactive interface for:
- Configuring communication strategy parameters
- Running simulations asynchronously
- Viewing belief alignment results
"""

import os
from pathlib import Path
from typing import Tuple

try:
    import gradio as gr
    _GRADIO_AVAILABLE = True
except ImportError:
    _GRADIO_AVAILABLE = False

from .strategies import StrategyConfig, CommunicationStrategy
from .simulation import CrisisSimulation
from .metrics import (
    results_to_rows,
    save_alignment_csv,
    plot_strategy_comparison,
    compute_summary_table,
    identify_winner,
)


def _make_strategy(
    name: str,
    start_timestep: int,
    tone_label: str,
    posts_per_timestep: int,
    multi_channel: bool,
) -> StrategyConfig:
    tone_map = {"authoritative": 1.0, "empathetic": 0.85, "neutral": 0.6, "reactive": 0.7}
    colors = {"authoritative": "#2196F3", "empathetic": "#4CAF50", "neutral": "#9E9E9E", "reactive": "#FF9800"}
    return StrategyConfig(
        name=name,
        start_timestep=int(start_timestep),
        tone_multiplier=tone_map.get(tone_label, 1.0),
        posts_per_timestep=int(posts_per_timestep),
        multi_channel=bool(multi_channel),
        tone_label=tone_label,
        color=colors.get(tone_label, "#607D8B"),
        description=f"Custom: {tone_label} tone, {posts_per_timestep}×/step, start t={start_timestep}",
    )


def run_simulation_sync(
    # Strategy 1
    s1_name: str, s1_start: int, s1_tone: str, s1_freq: int, s1_multi: bool,
    # Strategy 2
    s2_name: str, s2_start: int, s2_tone: str, s2_freq: int, s2_multi: bool,
    # Strategy 3
    s3_name: str, s3_start: int, s3_tone: str, s3_freq: int, s3_multi: bool,
    # Simulation params
    n_citizens: int, n_timesteps: int, seed: int,
) -> Tuple[str, str, str, str]:
    """Run simulation and return (status, plot_path, summary_md, csv_path)."""
    strategies = [
        _make_strategy(s1_name, s1_start, s1_tone, s1_freq, s1_multi),
        _make_strategy(s2_name, s2_start, s2_tone, s2_freq, s2_multi),
        _make_strategy(s3_name, s3_start, s3_tone, s3_freq, s3_multi),
    ]

    sim = CrisisSimulation(
        num_agents=int(n_citizens),
        num_timesteps=int(n_timesteps),
        num_misinfo_agents=int(os.getenv("NUM_MISINFO_AGENTS", "5")),
        seed=int(seed),
    )
    results = sim.run_comparison(strategies)
    rows = results_to_rows(results)

    csv_path = save_alignment_csv(rows)
    plot_path = plot_strategy_comparison(rows)
    summary_df = compute_summary_table(rows)
    winner = identify_winner(rows)

    status = f"Simulation complete! **Winner: {winner}**"
    return status, str(plot_path), summary_df.to_markdown(index=False), str(csv_path)


def create_ui() -> "gr.Blocks":
    """Build and return the Gradio Blocks app."""
    if not _GRADIO_AVAILABLE:
        raise ImportError("gradio is required. Install with: pip install gradio")

    defaults = CommunicationStrategy.all_defaults()
    d = {s.name: s for s in defaults}
    d1, d2, d3 = defaults[0], defaults[1], defaults[2]

    with gr.Blocks(title="OASIS Crisis Communication Optimizer", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
# OASIS Crisis Communication Optimizer
Model public health emergency communications. Configure competing strategies and measure
which approach maximizes public belief alignment with official guidance.
"""
        )

        with gr.Tabs():
            # ── Strategy Configuration Tab ──────────────────────────────────
            with gr.Tab("Strategy Configuration"):
                gr.Markdown("### Configure three competing communication strategies")

                all_strategy_inputs = []
                for i, (strat, color) in enumerate(zip(defaults, ["blue", "green", "red"])):
                    with gr.Accordion(f"Strategy {i+1}: {strat.name}", open=(i == 0)):
                        name = gr.Textbox(label="Strategy Name", value=strat.name)
                        start = gr.Slider(
                            label="Start Timestep",
                            minimum=1, maximum=15, step=1,
                            value=strat.start_timestep,
                        )
                        tone = gr.Radio(
                            label="Tone",
                            choices=["authoritative", "empathetic", "neutral", "reactive"],
                            value=strat.tone_label,
                        )
                        freq = gr.Slider(
                            label="Posts per Timestep",
                            minimum=1, maximum=5, step=1,
                            value=strat.posts_per_timestep,
                        )
                        multi = gr.Checkbox(
                            label="Multi-channel (amplifies reach)",
                            value=strat.multi_channel,
                        )
                        all_strategy_inputs.extend([name, start, tone, freq, multi])

                with gr.Row():
                    n_citizens = gr.Slider(
                        label="Citizens", minimum=20, maximum=200, step=10, value=50,
                    )
                    n_timesteps = gr.Slider(
                        label="Timesteps", minimum=10, maximum=50, step=5, value=20,
                    )
                    seed = gr.Number(label="Random Seed", value=42, precision=0)

                run_btn = gr.Button("Run Simulation", variant="primary", size="lg")
                status_out = gr.Markdown("Ready to run.")

            # ── Results Tab ─────────────────────────────────────────────────
            with gr.Tab("Results"):
                gr.Markdown("### Strategy Comparison")
                chart_out = gr.Image(label="Belief Alignment Over Time", type="filepath")
                summary_out = gr.Markdown(label="Summary Table")
                csv_out = gr.File(label="Download CSV")

        run_btn.click(
            fn=run_simulation_sync,
            inputs=all_strategy_inputs + [n_citizens, n_timesteps, seed],
            outputs=[status_out, chart_out, summary_out, csv_out],
        )

    return demo


def launch(share: bool = False, **kwargs):
    """Launch the Gradio UI."""
    app = create_ui()
    app.launch(share=share, **kwargs)


if __name__ == "__main__":
    launch()
