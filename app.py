"""
OASIS Crisis Communication Optimizer — Gradio UI

Allows users to configure and compare up to 4 communication strategies,
run simulations, and view results interactively.

Usage:
    python app.py [--port PORT] [--host HOST] [--model MODEL]
                  [--output-dir DIR] [--mock]
"""
from __future__ import annotations

import os
import io
import json
import logging
import argparse
import datetime
import time
import textwrap
from typing import Optional, List

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oasis_crisis_communi.strategies import StrategyConfig, CommunicationStrategy
from oasis_crisis_communi.simulation import CrisisSimulation, is_mock_mode
from oasis_crisis_communi.analyzer import ResultsAnalyzer
from oasis_crisis_communi.visualizer import SimulationVisualizer
from oasis_crisis_communi.reporter import ReportGenerator
from oasis_crisis_communi.scenarios import list_scenarios, get_scenario, ScenarioCard
from oasis_crisis_communi.sensitivity import compare_sensitivity

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_HOST = os.getenv("GRADIO_HOST", "0.0.0.0")
DEFAULT_MODEL = os.getenv("AGENT_MODEL", "openai/gpt-5.4-mini")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Available model choices for the picker dropdown
MODEL_CHOICES = [
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
    "minimax/minimax-m2.7",
    "mistralai/mistral-small-2603",
    "openai/gpt-5.4",
    "x-ai/grok-4.20-beta",
    "mock (no API key needed)",
]

TONE_MULTIPLIERS = {
    "authoritative": 1.0,
    "empathetic": 0.85,
    "reactive": 0.95,
    "neutral": 0.75,
}
TONE_COLORS = {
    "authoritative": "#2196F3",
    "empathetic": "#4CAF50",
    "reactive": "#FF9800",
    "neutral": "#9E9E9E",
}


def _validate_strategy_inputs(
    name: str,
    start: int,
    tone: str,
    freq: int,
    multi: bool,
) -> Optional[str]:
    """Return an error string if the strategy config is invalid, else None."""
    if not name.strip():
        return "Strategy name cannot be empty."
    if not (1 <= start <= 50):
        return f"Start timestep must be 1-50, got {start}."
    if tone not in TONE_MULTIPLIERS:
        return f"Unknown tone: {tone!r}. Must be one of {list(TONE_MULTIPLIERS)}."
    if not (1 <= freq <= 10):
        return f"Posts per timestep must be 1-10, got {freq}."
    return None


def build_strategy_from_ui(
    name: str,
    start_timestep: int,
    tone: str,
    posts_per_timestep: int,
    multi_channel: bool,
) -> StrategyConfig:
    """Build a StrategyConfig from Gradio UI inputs."""
    err = _validate_strategy_inputs(name, start_timestep, tone, posts_per_timestep, multi_channel)
    if err:
        raise ValueError(err)
    return StrategyConfig(
        name=name.strip(),
        start_timestep=int(start_timestep),
        tone_multiplier=TONE_MULTIPLIERS.get(tone, 0.85),
        posts_per_timestep=int(posts_per_timestep),
        multi_channel=bool(multi_channel),
        tone_label=tone,
        color=TONE_COLORS.get(tone, "#2196F3"),
        description=(
            f"Custom strategy: {tone} tone, starts at timestep {start_timestep}, "
            f"{posts_per_timestep} post(s)/step, multi-channel={multi_channel}"
        ),
    )


class _LogCapture(logging.Handler):
    """Capture log records into a list for UI display."""
    def __init__(self):
        super().__init__()
        self.lines: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        ts = datetime.datetime.utcfromtimestamp(record.created).strftime("%H:%M:%S")
        self.lines.append(f"[{ts}] {record.levelname}: {record.getMessage()}")


def run_simulation(
    num_agents: int,
    num_timesteps: int,
    num_misinfo: int,
    seed: int,
    model: str,
    # Strategy 1
    s1_name: str, s1_start: int, s1_tone: str, s1_freq: int, s1_multi: bool,
    # Strategy 2
    s2_name: str, s2_start: int, s2_tone: str, s2_freq: int, s2_multi: bool,
    # Strategy 3
    s3_name: str, s3_start: int, s3_tone: str, s3_freq: int, s3_multi: bool,
    # Strategy 4
    s4_name: str, s4_start: int, s4_tone: str, s4_freq: int, s4_multi: bool,
    # Which strategies are enabled
    use_s1: bool, use_s2: bool, use_s3: bool, use_s4: bool,
):
    """Run simulation and return results for Gradio UI."""
    # Attach log capture
    log_capture = _LogCapture()
    log_capture.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(log_capture)

    wall_start = time.monotonic()

    try:
        # Validate simulation params
        num_agents = int(num_agents)
        num_timesteps = int(num_timesteps)
        num_misinfo = int(num_misinfo)
        seed = int(seed)

        if not (1 <= num_agents <= 500):
            raise ValueError(f"num_agents must be 1-500, got {num_agents}")
        if not (1 <= num_timesteps <= 100):
            raise ValueError(f"num_timesteps must be 1-100, got {num_timesteps}")
        if not (0 <= num_misinfo <= 50):
            raise ValueError(f"num_misinfo must be 0-50, got {num_misinfo}")

        # Set model env var so downstream code can pick it up
        if model and "mock" not in model.lower():
            os.environ["AGENT_MODEL"] = model
        elif "mock" in model.lower():
            os.environ["FORCE_MOCK_MODE"] = "1"

        # Build enabled strategies
        strategies = []
        configs = [
            (use_s1, s1_name, s1_start, s1_tone, s1_freq, s1_multi),
            (use_s2, s2_name, s2_start, s2_tone, s2_freq, s2_multi),
            (use_s3, s3_name, s3_start, s3_tone, s3_freq, s3_multi),
            (use_s4, s4_name, s4_start, s4_tone, s4_freq, s4_multi),
        ]
        errors = []
        for idx, (enabled, name, start, tone, freq, multi) in enumerate(configs, 1):
            if not enabled or not str(name).strip():
                continue
            err = _validate_strategy_inputs(str(name), int(start), str(tone), int(freq), bool(multi))
            if err:
                errors.append(f"Strategy {idx}: {err}")
            else:
                strategies.append(build_strategy_from_ui(name, start, tone, freq, multi))

        if errors:
            err_msg = "Configuration errors:\n" + "\n".join(errors)
            return (_err(err_msg), None, None, None, gr.update(value=""), None, err_msg, "—", "—")

        if not strategies:
            msg = "No strategies enabled. Please enable at least one strategy."
            return (_err(msg), None, None, None, gr.update(value=""), None, msg, "—", "—")

        log_capture.lines.append(f"[{_ts()}] INFO: Starting simulation — "
                                  f"{len(strategies)} strategies, {num_agents} agents, "
                                  f"{num_timesteps} timesteps")

        sim = CrisisSimulation(
            num_agents=num_agents,
            num_timesteps=num_timesteps,
            num_misinfo_agents=num_misinfo,
            seed=seed,
        )
        results = sim.run_comparison(strategies)

        elapsed = time.monotonic() - wall_start
        log_capture.lines.append(f"[{_ts()}] INFO: Simulation complete in {elapsed:.2f}s")

        analyzer = ResultsAnalyzer(results)
        summary_df = analyzer.get_summary_table()
        recommendation = analyzer.generate_recommendation()
        run_stats = analyzer.get_run_stats()
        influencer_data = analyzer.compute_influencer_impact()

        # Generate charts
        viz = SimulationVisualizer(results, output_dir=OUTPUT_DIR)
        chart_path = viz.plot_belief_alignment()

        winner = analyzer.get_winner()
        belief_dist_path = None
        if winner:
            try:
                belief_dist_path = viz.plot_belief_distribution(winner)
                log_capture.lines.append(f"[{_ts()}] INFO: Belief distribution chart saved")
            except Exception as ve:
                log_capture.lines.append(f"[{_ts()}] WARNING: Could not plot belief distribution: {ve}")

        # Generate CSV
        csv_df = analyzer.to_dataframe()

        # Generate reports
        reporter = ReportGenerator(results, output_dir=OUTPUT_DIR)
        reporter.save_json_report()
        reporter.save_html_report()

        # Mode + timing display
        mock = is_mock_mode()
        mode_note = "⚠️ MOCK mode (no API key)" if mock else f"✅ Live mode — model: {model}"
        winner_text = ""
        if winner:
            winner_text = f"\n🏆 **Winner: {winner.strategy.name}** — {winner.final_alignment:.1%} alignment"

        # Token and timing stats
        total_tokens = sum(r.token_estimate for r in results)
        token_md = (
            f"**Token estimate:** {total_tokens:,}"
            + (" *(mock mode — 0 actual tokens used)*" if mock else "")
        )
        elapsed_md = f"**Elapsed:** {elapsed:.2f}s | **Strategies:** {len(strategies)} | **Agents:** {num_agents}"

        # Influencer info for status
        inf_notes = []
        for sname, info in influencer_data.items():
            inf_notes.append(
                f"  • {sname}: belief μ={info['belief_mean']:.2f} σ={info['belief_std']:.2f}, "
                f"{info['pct_above_60']:.0%} above 60%"
            )
        inf_section = ("\n\nPopulation Stats:\n" + "\n".join(inf_notes)) if inf_notes else ""

        status = (
            f"{mode_note}{winner_text}"
            f"\n\nSimulation complete — {len(strategies)} strategies, "
            f"{num_agents} agents, {num_timesteps} timesteps, {elapsed:.1f}s{inf_section}"
        )

        log_str = "\n".join(log_capture.lines[-30:])

        return (
            gr.update(value=status),
            chart_path,
            summary_df.to_html(index=False, classes="summary-table", border=0),
            csv_df,
            gr.update(value=recommendation),
            belief_dist_path,
            log_str,
            token_md,
            elapsed_md,
        )

    except ValueError as ve:
        msg = f"❌ Configuration error: {ve}"
        logger.warning(msg)
        return (_err(msg), None, None, None, gr.update(value=""), None, msg, "—", "—")
    except Exception as e:
        logger.exception("Simulation failed")
        msg = f"❌ Simulation error: {e}\n\nCheck logs for details."
        log_str = "\n".join(log_capture.lines)
        return (_err(msg), None, None, None, gr.update(value=""), None, log_str or msg, "—", "—")
    finally:
        logging.getLogger().removeHandler(log_capture)


def _err(msg: str):
    return gr.update(value=msg)


def _ts() -> str:
    return datetime.datetime.utcnow().strftime("%H:%M:%S")


def run_sensitivity_analysis(
    scenario_id: str,
    n_seeds: int,
    num_agents: int,
    num_timesteps: int,
):
    """Run sensitivity analysis for a scenario and return chart + summary."""
    try:
        n_seeds = int(n_seeds)
        num_agents = int(num_agents)
        num_timesteps = int(num_timesteps)

        if scenario_id == "custom":
            strats = CommunicationStrategy.all_defaults()
            label = "Default Strategies"
        else:
            scenario = get_scenario(scenario_id)
            strats = scenario.recommended_strategies
            label = scenario.name

        if n_seeds < 2:
            return None, "⚠️ n_seeds must be at least 2."

        sen_results = compare_sensitivity(
            strats,
            n_seeds=n_seeds,
            num_agents=num_agents,
            num_timesteps=num_timesteps,
        )

        viz_path = os.path.join(OUTPUT_DIR, "sensitivity_bands.png")
        import matplotlib.pyplot as _plt

        DEFAULT_STYLE = {
            "figure.facecolor": "#FAFAFA",
            "axes.facecolor": "#FFFFFF",
            "axes.grid": True,
            "grid.alpha": 0.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
        colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800"]
        with _plt.rc_context(DEFAULT_STYLE):
            fig, ax = _plt.subplots(figsize=(10, 6))
            for idx, sr in enumerate(sen_results):
                color = colors[idx % len(colors)]
                ts = list(range(1, len(sr.mean_belief_timeline) + 1))
                ax.plot(ts, sr.mean_belief_timeline, color=color, lw=2.5,
                        label=f"{sr.strategy_name}")
                ax.fill_between(ts, sr.ci_lower_95, sr.ci_upper_95,
                                color=color, alpha=0.18)
            ax.axhline(0.6, color="gray", lw=1, ls=":", alpha=0.7, label="60% threshold")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Mean Belief Alignment")
            ax.set_title(f"Robustness Analysis — {label}\n({n_seeds} seeds, shading = 95% CI)",
                         fontweight="bold")
            ax.set_ylim(0, 1)
            ax.legend(loc="lower right", fontsize=9)
            _plt.tight_layout()
            fig.savefig(viz_path, dpi=150, bbox_inches="tight")
            _plt.close(fig)

        # Build summary table
        rows = []
        for sr in sen_results:
            rows.append({
                "Strategy": sr.strategy_name,
                "Mean Final": f"{sr.final_alignment_mean:.1%}",
                "Std Dev": f"±{sr.final_alignment_std:.1%}",
                "Min": f"{sr.final_alignment_min:.1%}",
                "Max": f"{sr.final_alignment_max:.1%}",
                "CI Width": f"{sr.summary()['ci_width_final']:.1%}",
            })
        df = pd.DataFrame(rows)
        summary_md = f"### Sensitivity Results — {label} ({n_seeds} seeds)\n\n"
        summary_md += df.to_markdown(index=False)
        # Identify most robust strategy
        most_robust = min(sen_results, key=lambda r: r.final_alignment_std)
        best_mean = max(sen_results, key=lambda r: r.final_alignment_mean)
        summary_md += (
            f"\n\n**Most robust** (lowest variance): **{most_robust.strategy_name}** "
            f"(σ={most_robust.final_alignment_std:.3f})\n\n"
            f"**Best mean performance**: **{best_mean.strategy_name}** "
            f"(μ={best_mean.final_alignment_mean:.1%})"
        )
        return viz_path, summary_md

    except Exception as e:
        logger.exception("Sensitivity analysis failed")
        return None, f"❌ Sensitivity analysis error: {e}"


CSS = """
.summary-table { width: 100%; border-collapse: collapse; }
.summary-table th { background: #1a237e; color: white; padding: 8px 12px; text-align: left; }
.summary-table td { padding: 8px 12px; border-bottom: 1px solid #eee; }
.summary-table tr:hover td { background: #f5f5f5; }
.scenario-card { border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; margin-bottom: 8px; background: #fafafa; }
@media (max-width: 768px) {
    .gradio-row { flex-direction: column !important; }
    .gradio-row > * { width: 100% !important; min-width: unset !important; }
}
"""


def _make_load_scenario_fn(scenario_id: str, defaults):
    """Return a click handler that populates UI fields from a scenario."""
    def _handler():
        s = get_scenario(scenario_id)
        strats = s.recommended_strategies
        # Fill up to 4 strategy slots; pad with defaults for missing slots
        out = []
        for i in range(4):
            if i < len(strats):
                st = strats[i]
                out += [st.name, st.start_timestep, st.tone_label,
                        st.posts_per_timestep, st.multi_channel, True]
            else:
                out += [f"Unused {i+1}", 1, "neutral", 1, False, False]
        # Add sim params
        out += [
            s.sim_params.get("n_citizens", 50),
            s.sim_params.get("n_timesteps", 20),
            s.sim_params.get("n_misinfo_agents", 5),
        ]
        return out
    return _handler


def create_ui(model: Optional[str] = None) -> gr.Blocks:
    model = model or DEFAULT_MODEL

    with gr.Blocks(
        title="OASIS Crisis Communication Optimizer",
        theme=gr.themes.Soft(primary_hue="blue"),
        css=CSS,
    ) as demo:
        gr.Markdown("""
# 🏥 OASIS Crisis Communication Optimizer
**Simulate and compare public health emergency communication strategies**

Configure up to 4 strategies with different timing, tone, and frequency parameters.
The simulation measures which strategy achieves the highest belief alignment using
INTERVIEW actions on a population of synthetic social agents.

*Built autonomously by [NEO](https://heyneo.so) — your autonomous AI Agent* · [![Install NEO](https://img.shields.io/badge/VS%20Code-Install%20NEO-7B61FF?logo=visual-studio-code)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)
""")

        mock = is_mock_mode()
        mode_badge = (
            "⚠️ **MOCK MODE** — no API key detected. Results use simulated dynamics."
            if mock else "✅ **LIVE MODE** — LLM-powered agents active."
        )
        gr.Markdown(mode_badge)

        with gr.Tabs():
            # ═══════════════════════════════════════════════════════════════
            # Tab 1: Configure Strategies
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("⚙️ Configure Strategies"):
                with gr.Row():
                    num_agents = gr.Slider(10, 200, value=30, step=5, label="Public Agents")
                    num_timesteps = gr.Slider(5, 60, value=20, step=1, label="Simulation Timesteps")
                with gr.Row():
                    num_misinfo = gr.Slider(0, 20, value=5, step=1, label="Misinformation Agents")
                    seed = gr.Number(value=42, label="Random Seed", precision=0)
                with gr.Row():
                    model_picker = gr.Dropdown(
                        choices=MODEL_CHOICES,
                        value=model,
                        label="LLM Model (live mode only — ignored in mock mode)",
                    )

                gr.Markdown("---\n### Communication Strategies")
                gr.Markdown(
                    "Configure each strategy. Enable/disable with the checkbox. "
                    "Or load a pre-built scenario from the **Scenario Cards** tab."
                )

                strategy_inputs = []
                tone_radio_choices = [
                    ("authoritative — fact-based, official statements", "authoritative"),
                    ("empathetic — community-focused, trust-building", "empathetic"),
                    ("reactive — counter-misinformation focused", "reactive"),
                    ("neutral — balanced, informational", "neutral"),
                ]
                defaults = CommunicationStrategy.all_defaults()

                for i, (label_prefix, default_strategy) in enumerate([
                    ("Strategy 1", defaults[0]),
                    ("Strategy 2", defaults[1]),
                    ("Strategy 3", defaults[2]),
                    ("Strategy 4 (Custom)", None),
                ]):
                    with gr.Accordion(label_prefix, open=(i < 3)):
                        with gr.Row():
                            enabled = gr.Checkbox(value=(i < 3), label="Enable this strategy")
                            name = gr.Textbox(
                                value=default_strategy.name if default_strategy else f"Custom {i+1}",
                                label="Strategy Name",
                            )
                        with gr.Row():
                            start = gr.Slider(
                                1, 30,
                                value=default_strategy.start_timestep if default_strategy else 1,
                                step=1, label="Start Timestep (1 = immediately)",
                            )
                            freq = gr.Slider(
                                1, 5,
                                value=default_strategy.posts_per_timestep if default_strategy else 1,
                                step=1, label="Posts per Timestep",
                            )
                        tone = gr.Radio(
                            choices=tone_radio_choices,
                            value=default_strategy.tone_label if default_strategy else "neutral",
                            label="Messaging Tone",
                        )
                        multi = gr.Checkbox(
                            value=default_strategy.multi_channel if default_strategy else False,
                            label="Multi-channel (amplifies reach across platforms)",
                        )
                        strategy_inputs.extend([name, start, tone, freq, multi, enabled])

                run_btn = gr.Button("▶ Run Simulation", variant="primary", size="lg")

            # ═══════════════════════════════════════════════════════════════
            # Tab 2: Results
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("📊 Results"):
                status_box = gr.Textbox(
                    label="Simulation Status",
                    value="Configure strategies and click 'Run Simulation'",
                    lines=5,
                    interactive=False,
                )
                with gr.Row():
                    token_display = gr.Markdown("**Token estimate:** —")
                    elapsed_display = gr.Markdown("**Elapsed / Strategies / Agents:** —")

                comparison_chart = gr.Image(
                    label="Strategy Comparison Chart", type="filepath"
                )
                belief_dist_chart = gr.Image(
                    label="Final Belief Distribution (Winning Strategy)", type="filepath"
                )
                summary_html = gr.HTML(label="Summary Table")
                log_box = gr.Textbox(
                    label="Run Log",
                    lines=6,
                    interactive=False,
                    placeholder="Simulation log will appear here after running...",
                )

            # ═══════════════════════════════════════════════════════════════
            # Tab 3: Data & Download
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("📋 Data & Download"):
                recommendation_box = gr.Textbox(
                    label="Strategy Recommendation",
                    lines=22,
                    interactive=False,
                )
                alignment_data = gr.Dataframe(
                    label="Belief Alignment Data",
                    headers=["strategy", "timestep", "alignment_score",
                             "repost_rate", "misinfo_repost_rate"],
                )
                gr.Markdown(
                    "After running, download full reports from the `outputs/` directory:\n"
                    "- `outputs/results.json` — full JSON report\n"
                    "- `outputs/report.html` — styled HTML report\n"
                    "- `outputs/belief_alignment.csv` — raw data"
                )

            # ═══════════════════════════════════════════════════════════════
            # Tab 4: Scenario Cards
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("🎬 Scenario Cards"):
                gr.Markdown("""
### Pre-built Crisis Scenarios

Click **Load Scenario** on any card to instantly configure the strategy
parameters and simulation settings for that crisis type.
Then switch to **Configure Strategies** and click **Run Simulation**.
""")
                scenario_outputs = (
                    [strategy_inputs[i * 6 + j] for i in range(4) for j in range(6)]
                    + [num_agents, num_timesteps, num_misinfo]
                )

                for scenario in list_scenarios():
                    severity_color = {
                        "critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"
                    }.get(scenario.severity, "⚪")
                    with gr.Group():
                        gr.Markdown(
                            f"## {scenario.icon} {scenario.name}\n"
                            f"{severity_color} **{scenario.severity.upper()}** · "
                            f"_{scenario.crisis_type}_\n\n"
                            f"{scenario.description}\n\n"
                            + "**Strategies:** "
                            + " · ".join(
                                f"`{s.name}` ({s.tone_label})"
                                for s in scenario.recommended_strategies
                            )
                            + f"\n\n**Suggested params:** "
                            f"{scenario.sim_params['n_citizens']} agents, "
                            f"{scenario.sim_params['n_timesteps']} timesteps, "
                            f"{scenario.sim_params['n_misinfo_agents']} misinfo agents"
                        )
                        load_btn = gr.Button(
                            f"📥 Load: {scenario.name}", variant="secondary"
                        )
                        load_fn = _make_load_scenario_fn(scenario.id, defaults)
                        load_btn.click(
                            fn=load_fn,
                            inputs=[],
                            outputs=scenario_outputs,
                        )
                    gr.Markdown("---")

            # ═══════════════════════════════════════════════════════════════
            # Tab 5: Sensitivity Analysis
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("📈 Sensitivity Analysis"):
                gr.Markdown("""
### Strategy Robustness Analysis

Run any scenario across multiple random seeds to compute **mean alignment ± 95% CI**
for each strategy. This reveals how reliable a strategy's performance is across
different population conditions — not just one lucky seed.
""")
                with gr.Row():
                    sen_scenario = gr.Dropdown(
                        choices=[("Default (3 built-in strategies)", "custom")]
                        + [(s.name, s.id) for s in list_scenarios()],
                        value="custom",
                        label="Scenario",
                    )
                    sen_n_seeds = gr.Slider(5, 50, value=15, step=5, label="Number of Seeds")
                with gr.Row():
                    sen_agents = gr.Slider(10, 100, value=30, step=5, label="Agents per Run")
                    sen_timesteps = gr.Slider(5, 40, value=20, step=5, label="Timesteps per Run")

                sen_run_btn = gr.Button("▶ Run Sensitivity Analysis", variant="primary")
                sen_chart = gr.Image(label="Robustness — Mean ± 95% CI", type="filepath")
                sen_summary = gr.Markdown("Summary will appear here after analysis.")

                sen_run_btn.click(
                    fn=run_sensitivity_analysis,
                    inputs=[sen_scenario, sen_n_seeds, sen_agents, sen_timesteps],
                    outputs=[sen_chart, sen_summary],
                )

            # ═══════════════════════════════════════════════════════════════
            # Tab 6: About
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("ℹ️ About"):
                gr.Markdown(textwrap.dedent("""
## About OASIS Crisis Communication Optimizer

This tool simulates public health emergency communications using the CAMEL-AI OASIS
framework, letting you compare strategy effectiveness before real-world deployment.

### How It Works

1. **Government Agent**: Posts official guidance following the configured strategy
2. **Misinformation Agents**: Post contradictory content to undermine trust
3. **Public Agents**: Read posts, influence each other, update their beliefs
4. **INTERVIEW Actions**: At each timestep, agents are queried about their beliefs
5. **Alignment Score**: Fraction of population aligned with official guidance (0-1)

### Strategy Parameters

| Parameter | Description |
|-----------|-------------|
| **Start Timestep** | When government begins posting (1=immediately, 8=delayed) |
| **Tone** | authoritative=fact-based, empathetic=community-focused, reactive=counter-misinfo |
| **Posts per Timestep** | Frequency of government communications |
| **Multi-channel** | Amplifies reach across multiple platforms |

### New Features

| Feature | Description |
|---------|-------------|
| **Scenario Cards** | Pre-built COVID, outbreak, disaster, bioterrorism scenarios |
| **Sensitivity Analysis** | Multi-seed robustness testing with 95% CI bands |
| **Belief Distribution** | Histogram of final agent belief spread for winning strategy |
| **Token Counter** | Estimated LLM token usage per run |
| **Timing Stats** | Wall-clock time for each simulation run |
| **Model Picker** | Select from latest OpenRouter models |

### Research Background

Based on the OASIS paper: *"OASIS: Open Agent Social Interaction Simulations"*.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | API key for LLM mode |
| `AGENT_MODEL` | `openai/gpt-5.4-mini` | Default LLM model |
| `FORCE_MOCK_MODE` | `0` | Force mock mode even with API key |
| `NUM_AGENTS` | `30` | Default agent count |
| `NUM_TIMESTEPS` | `20` | Default timestep count |
| `RANDOM_SEED` | `42` | Default random seed |
| `OUTPUT_DIR` | `outputs` | Output directory |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `GRADIO_PORT` | `7860` | UI server port |

### Made With
- [CAMEL-AI OASIS](https://github.com/camel-ai/oasis) — Social simulation framework
- [Gradio](https://gradio.app) — UI framework
- Made autonomously using [NEO](https://heyneo.so)
"""))

        # ── Wire run button ────────────────────────────────────────────────
        s_names   = [strategy_inputs[i * 6 + 0] for i in range(4)]
        s_starts  = [strategy_inputs[i * 6 + 1] for i in range(4)]
        s_tones   = [strategy_inputs[i * 6 + 2] for i in range(4)]
        s_freqs   = [strategy_inputs[i * 6 + 3] for i in range(4)]
        s_multis  = [strategy_inputs[i * 6 + 4] for i in range(4)]
        s_enabled = [strategy_inputs[i * 6 + 5] for i in range(4)]

        run_inputs = [
            num_agents, num_timesteps, num_misinfo, seed, model_picker,
            s_names[0], s_starts[0], s_tones[0], s_freqs[0], s_multis[0],
            s_names[1], s_starts[1], s_tones[1], s_freqs[1], s_multis[1],
            s_names[2], s_starts[2], s_tones[2], s_freqs[2], s_multis[2],
            s_names[3], s_starts[3], s_tones[3], s_freqs[3], s_multis[3],
            s_enabled[0], s_enabled[1], s_enabled[2], s_enabled[3],
        ]

        run_outputs = [
            status_box,
            comparison_chart,
            summary_html,
            alignment_data,
            recommendation_box,
            belief_dist_chart,
            log_box,
            token_display,
            elapsed_display,
        ]

        run_btn.click(fn=run_simulation, inputs=run_inputs, outputs=run_outputs)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OASIS Crisis Communication Optimizer")
    parser.add_argument("--port", type=int, default=GRADIO_PORT,
                        help="Gradio server port (default: 7860)")
    parser.add_argument("--host", type=str, default=GRADIO_HOST,
                        help="Gradio server host (default: 0.0.0.0)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="LLM model ID for live mode")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Directory to save outputs")
    parser.add_argument("--mock", action="store_true",
                        help="Force mock mode (no API calls)")
    args = parser.parse_args()

    if args.mock:
        os.environ["FORCE_MOCK_MODE"] = "1"
    if args.output_dir != OUTPUT_DIR:
        os.environ["OUTPUT_DIR"] = args.output_dir
        os.makedirs(args.output_dir, exist_ok=True)

    demo = create_ui(model=args.model)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        show_error=True,
    )
