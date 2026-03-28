"""Pytest fixtures for OASIS Crisis Communication Optimizer tests."""
import os
import pytest

# Ensure we're in mock mode for tests (no API calls)
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("OPENROUTER_API_KEY", "")
os.environ.setdefault("NUM_AGENTS", "10")
os.environ.setdefault("NUM_TIMESTEPS", "5")
os.environ.setdefault("RANDOM_SEED", "42")

from oasis_crisis_communi.strategies import StrategyConfig, CommunicationStrategy
from oasis_crisis_communi.simulation import CrisisSimulation, MockSimulation
from oasis_crisis_communi.belief_tracker import BeliefTracker
from oasis_crisis_communi.analyzer import ResultsAnalyzer


@pytest.fixture
def small_strategy():
    """A minimal strategy for fast testing."""
    return StrategyConfig(
        name="Test Strategy",
        start_timestep=1,
        tone_multiplier=1.0,
        posts_per_timestep=1,
        multi_channel=False,
        tone_label="authoritative",
        color="#2196F3",
        description="Test strategy",
    )


@pytest.fixture
def default_strategies():
    """All three default strategies."""
    return CommunicationStrategy.all_defaults()


@pytest.fixture
def small_simulation(small_strategy):
    """A fast mock simulation with few agents/timesteps."""
    return MockSimulation(
        strategy=small_strategy,
        num_agents=10,
        num_timesteps=5,
        num_misinfo_agents=2,
        seed=42,
    )


@pytest.fixture
def simulation_result(small_simulation):
    """Pre-run simulation result."""
    return small_simulation.run()


@pytest.fixture
def belief_tracker():
    """Fresh BeliefTracker instance."""
    return BeliefTracker()


@pytest.fixture
def three_results(default_strategies, tmp_path):
    """Three strategy results for comparison tests."""
    sim = CrisisSimulation(
        num_agents=10,
        num_timesteps=5,
        num_misinfo_agents=2,
        seed=42,
    )
    return sim.run_comparison(default_strategies)
