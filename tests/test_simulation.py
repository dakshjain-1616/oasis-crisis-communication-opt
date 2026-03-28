"""
Tests for the OASIS Crisis Communication Optimizer.

Test spec:
1. demo.py completes 3-strategy simulation without error
2. outputs/belief_alignment.csv has columns: strategy, timestep, alignment_score (0-1)
3. outputs/strategy_comparison.png shows 3 curves over time
4. INTERVIEW action correctly parses agent belief (returns a structured response)
5. Winning strategy identified with score > 0.5 alignment
6. Gradio UI: strategy params configurable, async run works, results tab shows comparison chart
"""

import sys
import subprocess
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from oasis_crisis_communi.strategies import StrategyConfig, CommunicationStrategy
from oasis_crisis_communi.simulation import CrisisSimulation, SimulationResult, MockSimulation
from oasis_crisis_communi.belief_tracker import BeliefTracker, InterviewResponse
from oasis_crisis_communi.metrics import (
    results_to_rows,
    save_alignment_csv,
    plot_strategy_comparison,
    compute_summary_table,
    identify_winner,
)
from oasis_crisis_communi.mock_oasis import ActionType, SocialAgent
from oasis_crisis_communi.mock_oasis import InterviewResponse as MoIR


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def default_strategies():
    return CommunicationStrategy.all_defaults()


@pytest.fixture(scope="session")
def simulation_results(default_strategies):
    """Run the 3-strategy comparison once, shared across tests."""
    sim = CrisisSimulation(num_agents=30, num_timesteps=15, num_misinfo_agents=3, seed=42)
    return sim.run_comparison(default_strategies)


@pytest.fixture(scope="session")
def flat_rows(simulation_results):
    return results_to_rows(simulation_results)


@pytest.fixture(scope="session")
def saved_csv(flat_rows, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("outputs")
    return save_alignment_csv(flat_rows, output_path=tmp / "belief_alignment.csv")


@pytest.fixture(scope="session")
def saved_plot(flat_rows, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("plots")
    return plot_strategy_comparison(flat_rows, output_path=tmp / "strategy_comparison.png")


# ── Test 1: demo.py completes without error ───────────────────────────────────

def test_demo_runs_without_error():
    """demo.py must complete 3-strategy simulation without raising errors."""
    demo_script = ROOT / "scripts" / "demo.py"
    result = subprocess.run(
        [sys.executable, str(demo_script)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"demo.py exited with code {result.returncode}\n"
        f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
    )


# ── Test 2: CSV has correct columns and valid data ────────────────────────────

def test_csv_columns(saved_csv):
    """CSV must have columns: strategy, timestep, alignment_score."""
    import pandas as pd
    df = pd.read_csv(saved_csv)
    assert set(df.columns) == {"strategy", "timestep", "alignment_score"}, (
        f"Unexpected columns: {list(df.columns)}"
    )


def test_csv_alignment_score_range(saved_csv):
    """alignment_score values must be in [0, 1]."""
    import pandas as pd
    df = pd.read_csv(saved_csv)
    assert (df["alignment_score"] >= 0).all(), "alignment_score contains values < 0"
    assert (df["alignment_score"] <= 1).all(), "alignment_score contains values > 1"


def test_csv_has_three_strategies(saved_csv):
    """CSV must contain rows for exactly 3 strategies."""
    import pandas as pd
    df = pd.read_csv(saved_csv)
    assert df["strategy"].nunique() == 3, (
        f"Expected 3 strategies, got {df['strategy'].nunique()}: {list(df['strategy'].unique())}"
    )


def test_csv_not_empty(saved_csv):
    """CSV must have data rows."""
    import pandas as pd
    df = pd.read_csv(saved_csv)
    assert len(df) > 0


# ── Test 3: Plot file exists and is valid ─────────────────────────────────────

def test_plot_file_exists(saved_plot):
    assert saved_plot.exists(), f"Plot file not found at {saved_plot}"


def test_plot_is_valid_png(saved_plot):
    PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
    with open(saved_plot, "rb") as f:
        header = f.read(8)
    assert header == PNG_MAGIC, "Plot file is not a valid PNG"


def test_plot_has_three_curves(flat_rows):
    """Plot data must contain 3 distinct strategy curves."""
    import pandas as pd
    df = pd.DataFrame(flat_rows)
    assert df["strategy"].nunique() == 3


# ── Test 4: INTERVIEW action returns structured response ──────────────────────

def test_interview_returns_structured_response():
    """BeliefTracker.parse_interview_response() must return a structured InterviewResponse."""
    tracker = BeliefTracker()
    response = tracker.parse_interview_response(
        agent_id=1,
        timestep=5,
        raw_response=(
            "I trust the official guidance completely. The government has been "
            "transparent and the evidence is credible."
        ),
    )
    assert isinstance(response, InterviewResponse), (
        f"Expected InterviewResponse, got {type(response)}"
    )
    assert 0.0 <= response.belief_score <= 1.0, (
        f"belief_score {response.belief_score} out of [0,1]"
    )
    assert response.sentiment in ("positive", "neutral", "negative"), (
        f"unexpected sentiment: {response.sentiment}"
    )
    assert isinstance(response.reasoning if hasattr(response, 'reasoning') else
                      response.raw_response, str)


def test_interview_to_dict():
    """InterviewResponse.to_dict() must include agent_id and belief_score."""
    tracker = BeliefTracker()
    response = tracker.parse_interview_response(
        agent_id=7,
        timestep=3,
        raw_response="I'm uncertain about the official guidance.",
    )
    d = response.to_dict()
    assert "agent_id" in d
    assert "belief_score" in d
    assert d["agent_id"] == 7


def test_interview_positive_trust():
    """Strongly positive response should have belief_score > 0.5."""
    tracker = BeliefTracker()
    response = tracker.parse_interview_response(
        agent_id=2,
        timestep=1,
        raw_response=(
            "I trust the official guidance. I believe their recommendations "
            "are accurate and helpful. I agree with and support following them."
        ),
    )
    assert response.belief_score > 0.5, (
        f"Positive response got belief_score={response.belief_score}"
    )
    assert response.sentiment == "positive"


def test_interview_negative_distrust():
    """Distrust response should have belief_score < 0.5."""
    tracker = BeliefTracker()
    response = tracker.parse_interview_response(
        agent_id=3,
        timestep=1,
        raw_response=(
            "I distrust the government completely. I don't believe their "
            "claims are accurate. The guidance is false propaganda."
        ),
    )
    assert response.belief_score < 0.5, (
        f"Negative response got belief_score={response.belief_score}"
    )
    assert response.sentiment == "negative"


def test_interview_action_type_enum():
    """ActionType.INTERVIEW must exist."""
    assert ActionType.INTERVIEW is not None
    assert ActionType.INTERVIEW.value == "interview"


def test_mock_oasis_interview():
    """mock_oasis.SocialAgent.interview() also returns a structured response."""
    agent = SocialAgent(agent_id=5, initial_belief=0.75)
    response = agent.interview()
    assert isinstance(response, MoIR)
    assert 0.0 <= response.belief_score <= 1.0
    assert isinstance(response.reasoning, str) and len(response.reasoning) > 0


# ── Test 5: Winning strategy has score > 0.5 ─────────────────────────────────

def test_winning_strategy_score_above_threshold(simulation_results, flat_rows):
    """Best-performing strategy must achieve final alignment > 0.5."""
    winner_result = max(simulation_results, key=lambda r: r.final_alignment)
    assert winner_result.final_alignment > 0.5, (
        f"Winning strategy '{winner_result.strategy.name}' has final score "
        f"{winner_result.final_alignment:.4f}, expected > 0.5"
    )


def test_winner_identified(flat_rows):
    """identify_winner() must return a non-empty string."""
    winner = identify_winner(flat_rows)
    assert isinstance(winner, str) and len(winner) > 0


def test_early_strategy_beats_late(simulation_results):
    """Early & Authoritative should outperform Late & Reactive."""
    scores = {r.strategy.name: r.final_alignment for r in simulation_results}
    early = scores.get("Early & Authoritative", 0)
    late = scores.get("Late & Reactive", 0)
    assert early >= late, (
        f"Early & Authoritative ({early:.4f}) should >= Late & Reactive ({late:.4f})"
    )


def test_simulation_result_structure(simulation_results):
    """Each SimulationResult must have required fields."""
    for result in simulation_results:
        assert isinstance(result, SimulationResult)
        assert len(result.belief_timeline) > 0
        assert all(0.0 <= s <= 1.0 for s in result.belief_timeline), (
            f"belief_timeline out of [0,1] for {result.strategy.name}"
        )
        assert len(result.repost_rates) == len(result.belief_timeline)
        assert len(result.misinfo_repost_rates) == len(result.belief_timeline)


def test_exactly_one_winner(simulation_results):
    """Exactly one result should be marked as winning."""
    winners = [r for r in simulation_results if r.winning]
    assert len(winners) == 1, f"Expected 1 winner, got {len(winners)}"


# ── Test 6: Gradio UI ─────────────────────────────────────────────────────────

def test_gradio_ui_importable():
    """oasis_crisis_communi.ui must be importable."""
    from oasis_crisis_communi import ui
    assert ui is not None


def test_gradio_create_ui():
    """create_ui() must return a Gradio Blocks object if gradio is installed."""
    pytest.importorskip("gradio", reason="gradio not installed")
    from oasis_crisis_communi.ui import create_ui
    import gradio as gr
    app = create_ui()
    assert isinstance(app, gr.Blocks), f"Expected gr.Blocks, got {type(app)}"


def test_gradio_strategy_params_configurable():
    """run_simulation_sync must accept custom strategy params and return results."""
    from oasis_crisis_communi.ui import run_simulation_sync
    status, plot_path, summary_md, csv_path = run_simulation_sync(
        "TestA", 1, "authoritative", 2, False,
        "TestB", 1, "empathetic", 3, True,
        "TestC", 8, "reactive", 1, False,
        30, 10, 99,
    )
    assert "Winner" in status, f"Status should mention winner: {status}"
    assert str(plot_path).endswith(".png"), f"Expected PNG path: {plot_path}"
    assert str(csv_path).endswith(".csv"), f"Expected CSV path: {csv_path}"
    assert len(summary_md) > 0, "Summary markdown should not be empty"


def test_gradio_async_run():
    """run_simulation_sync must be callable and return 4 outputs."""
    from oasis_crisis_communi.ui import run_simulation_sync
    results = run_simulation_sync(
        "S1", 1, "authoritative", 1, False,
        "S2", 2, "empathetic", 2, True,
        "S3", 8, "neutral", 1, False,
        20, 10, 7,
    )
    assert len(results) == 4, f"Expected 4 outputs, got {len(results)}"


def test_gradio_results_show_comparison_chart():
    """The chart output from run_simulation_sync must be a valid file path."""
    from oasis_crisis_communi.ui import run_simulation_sync
    from pathlib import Path
    _, plot_path, _, _ = run_simulation_sync(
        "A", 1, "authoritative", 2, False,
        "B", 1, "empathetic", 2, True,
        "C", 5, "neutral", 1, False,
        20, 10, 0,
    )
    p = Path(plot_path)
    assert p.exists(), f"Chart file does not exist: {plot_path}"
    assert p.suffix == ".png", f"Expected .png, got {p.suffix}"


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_strategy_config_fields():
    """StrategyConfig must have required fields with valid values."""
    for strat in CommunicationStrategy.all_defaults():
        assert isinstance(strat.name, str) and len(strat.name) > 0
        assert 1 <= strat.start_timestep
        assert 0.0 < strat.tone_multiplier <= 1.0
        assert 1 <= strat.posts_per_timestep <= 10
        assert isinstance(strat.multi_channel, bool)
        assert isinstance(strat.color, str)


def test_mock_simulation_scores_in_range():
    """MockSimulation belief_timeline values must be in [0,1]."""
    strat = CommunicationStrategy.early_authoritative()
    sim = MockSimulation(strategy=strat, num_agents=10, num_timesteps=5, seed=0)
    result = sim.run()
    for t, score in enumerate(result.belief_timeline):
        assert 0.0 <= score <= 1.0, f"Score {score} at t={t} out of [0,1]"


def test_mock_simulation_correct_length():
    """MockSimulation must produce one score per timestep."""
    strat = CommunicationStrategy.late_reactive()
    sim = MockSimulation(strategy=strat, num_agents=10, num_timesteps=8, seed=0)
    result = sim.run()
    assert len(result.belief_timeline) == 8


def test_crisis_simulation_run_comparison():
    """CrisisSimulation.run_comparison must return N results for N strategies."""
    strategies = CommunicationStrategy.all_defaults()
    sim = CrisisSimulation(num_agents=10, num_timesteps=5, seed=0)
    results = sim.run_comparison(strategies)
    assert len(results) == len(strategies)


def test_results_to_rows_format(simulation_results):
    """results_to_rows must produce dicts with required keys."""
    rows = results_to_rows(simulation_results)
    assert len(rows) > 0
    for row in rows[:5]:
        assert "strategy" in row
        assert "timestep" in row
        assert "alignment_score" in row
        assert 0.0 <= row["alignment_score"] <= 1.0


def test_belief_tracker_population_alignment():
    """BeliefTracker.get_population_alignment must return value in [0,1]."""
    tracker = BeliefTracker()
    for i in range(5):
        tracker.parse_interview_response(
            agent_id=i, timestep=1,
            raw_response="I trust and believe the official guidance is accurate.",
        )
    score = tracker.get_population_alignment(timestep=1)
    assert 0.0 <= score <= 1.0, f"Population alignment {score} out of [0,1]"


def test_belief_tracker_agent_trajectory():
    """BeliefTracker should track per-agent belief trajectory."""
    tracker = BeliefTracker()
    tracker.parse_interview_response(
        agent_id=42, timestep=1,
        raw_response="I trust the guidance.",
    )
    tracker.parse_interview_response(
        agent_id=42, timestep=2,
        raw_response="I believe the official information is correct.",
    )
    traj = tracker.get_agent_belief_trajectory(42)
    assert len(traj) == 2
