"""Tests for Scenario Cards (oasis_crisis_communi/scenarios.py)."""
import pytest

from oasis_crisis_communi.scenarios import (
    ALL_SCENARIOS,
    SCENARIO_COVID_PANDEMIC,
    SCENARIO_DISEASE_OUTBREAK,
    SCENARIO_NATURAL_DISASTER,
    SCENARIO_BIOTERRORISM,
    ScenarioCard,
    get_scenario,
    list_scenarios,
)
from oasis_crisis_communi.strategies import StrategyConfig
from oasis_crisis_communi.simulation import CrisisSimulation


# ── Inventory tests ───────────────────────────────────────────────────────────

def test_four_scenarios_defined():
    assert len(list_scenarios()) == 4


def test_scenario_ids_unique():
    ids = [s.id for s in list_scenarios()]
    assert len(ids) == len(set(ids)), f"Duplicate scenario IDs: {ids}"


def test_all_scenario_ids_present():
    expected = {"covid_pandemic", "disease_outbreak", "natural_disaster", "bioterrorism"}
    assert set(ALL_SCENARIOS.keys()) == expected


# ── ScenarioCard field validation ─────────────────────────────────────────────

def test_scenario_fields_non_empty():
    for s in list_scenarios():
        assert isinstance(s.id, str) and s.id
        assert isinstance(s.name, str) and s.name
        assert isinstance(s.description, str) and len(s.description) > 10
        assert s.crisis_type in {"pandemic", "outbreak", "natural_disaster", "bioterrorism"}
        assert s.severity in {"low", "medium", "high", "critical"}


def test_scenario_has_strategies():
    for s in list_scenarios():
        assert len(s.recommended_strategies) >= 1, f"{s.id} has no strategies"
        for strat in s.recommended_strategies:
            assert isinstance(strat, StrategyConfig), (
                f"{s.id} strategy is not a StrategyConfig: {type(strat)}"
            )


def test_scenario_sim_params():
    for s in list_scenarios():
        assert "n_citizens" in s.sim_params, f"{s.id} missing n_citizens"
        assert "n_timesteps" in s.sim_params, f"{s.id} missing n_timesteps"
        assert "n_misinfo_agents" in s.sim_params, f"{s.id} missing n_misinfo_agents"
        assert s.sim_params["n_citizens"] > 0
        assert s.sim_params["n_timesteps"] > 0
        assert s.sim_params["n_misinfo_agents"] >= 0


# ── Strategy validity ─────────────────────────────────────────────────────────

def test_scenario_strategies_valid():
    for s in list_scenarios():
        for strat in s.recommended_strategies:
            assert strat.start_timestep >= 1, (
                f"{s.id}/{strat.name}: start_timestep must be >= 1"
            )
            assert 0.0 < strat.tone_multiplier <= 1.0, (
                f"{s.id}/{strat.name}: tone_multiplier out of (0, 1]"
            )
            assert 1 <= strat.posts_per_timestep <= 10, (
                f"{s.id}/{strat.name}: posts_per_timestep out of range"
            )
            assert strat.tone_label in {
                "authoritative", "empathetic", "reactive", "neutral"
            }, f"{s.id}/{strat.name}: unknown tone_label"


def test_scenario_strategy_names_unique_within_scenario():
    for s in list_scenarios():
        names = [st.name for st in s.recommended_strategies]
        assert len(names) == len(set(names)), (
            f"{s.id} has duplicate strategy names: {names}"
        )


# ── Lookup helpers ────────────────────────────────────────────────────────────

def test_get_scenario_by_id():
    s = get_scenario("covid_pandemic")
    assert isinstance(s, ScenarioCard)
    assert s.id == "covid_pandemic"


def test_get_scenario_unknown_raises():
    with pytest.raises(KeyError, match="nonexistent"):
        get_scenario("nonexistent")


def test_list_scenarios_returns_all():
    scenarios = list_scenarios()
    assert len(scenarios) == len(ALL_SCENARIOS)
    ids_from_list = {s.id for s in scenarios}
    assert ids_from_list == set(ALL_SCENARIOS.keys())


# ── Domain-logic checks ───────────────────────────────────────────────────────

def test_covid_scenario_has_multi_channel():
    """COVID pandemic should recommend at least one multi-channel strategy."""
    s = get_scenario("covid_pandemic")
    assert any(st.multi_channel for st in s.recommended_strategies), (
        "COVID scenario should include a multi-channel strategy"
    )


def test_bioterrorism_has_critical_severity():
    assert SCENARIO_BIOTERRORISM.severity == "critical"


def test_bioterrorism_has_most_misinfo_agents():
    """Bioterrorism scenario should have the most misinfo agents."""
    max_misinfo = max(s.sim_params["n_misinfo_agents"] for s in list_scenarios())
    assert SCENARIO_BIOTERRORISM.sim_params["n_misinfo_agents"] == max_misinfo


def test_natural_disaster_short_timesteps():
    """Natural disaster requires rapid response — shortest or tied-shortest run."""
    min_ts = min(s.sim_params["n_timesteps"] for s in list_scenarios())
    assert SCENARIO_NATURAL_DISASTER.sim_params["n_timesteps"] == min_ts


# ── Integration: scenario strategies run without error ────────────────────────

def test_scenario_strategies_run_in_simulation():
    """Each scenario's first recommended strategy must run through MockSimulation."""
    for s in list_scenarios():
        strat = s.recommended_strategies[0]
        sim = CrisisSimulation(
            strategy=strat,
            n_citizens=15,
            n_timesteps=5,
            n_misinfo_agents=2,
            seed=0,
        )
        scores = sim.run()
        assert len(scores) == 5, f"{s.id}: expected 5 scores, got {len(scores)}"
        assert all(0.0 <= v <= 1.0 for v in scores), (
            f"{s.id}: scores out of [0,1]: {scores}"
        )


def test_scenario_comparison_marks_winner():
    """run_comparison on scenario strategies should identify exactly one winner."""
    s = get_scenario("disease_outbreak")
    sim = CrisisSimulation(
        n_citizens=15,
        n_timesteps=5,
        n_misinfo_agents=2,
        seed=42,
    )
    results = sim.run_comparison(s.recommended_strategies)
    winners = [r for r in results if r.winning]
    assert len(winners) == 1, f"Expected 1 winner, got {len(winners)}"
