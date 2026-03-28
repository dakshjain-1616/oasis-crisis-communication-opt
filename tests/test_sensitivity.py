"""Tests for multi-seed sensitivity analysis (oasis_crisis_communi/sensitivity.py)."""
import pytest
import numpy as np

from oasis_crisis_communi.sensitivity import (
    SensitivityResult,
    compare_sensitivity,
    run_sensitivity,
)
from oasis_crisis_communi.strategies import CommunicationStrategy


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def early_strat():
    return CommunicationStrategy.early_authoritative()


@pytest.fixture(scope="module")
def all_strats():
    return CommunicationStrategy.all_defaults()


# ── Basic structure ───────────────────────────────────────────────────────────

def test_sensitivity_returns_correct_type(early_strat):
    result = run_sensitivity(early_strat, n_seeds=5, num_agents=10, num_timesteps=5)
    assert isinstance(result, SensitivityResult)


def test_sensitivity_timeline_length(early_strat):
    n_ts = 7
    result = run_sensitivity(early_strat, n_seeds=5, num_agents=10, num_timesteps=n_ts)
    assert len(result.mean_belief_timeline) == n_ts
    assert len(result.ci_lower_95) == n_ts
    assert len(result.ci_upper_95) == n_ts
    assert len(result.std_timeline) == n_ts


def test_sensitivity_seed_count(early_strat):
    result = run_sensitivity(early_strat, n_seeds=8, num_agents=10, num_timesteps=5)
    assert result.n_seeds == 8
    assert len(result.seeds) == 8


def test_sensitivity_seed_sequence(early_strat):
    result = run_sensitivity(early_strat, n_seeds=5, base_seed=10,
                             num_agents=10, num_timesteps=5)
    assert result.seeds == list(range(10, 15))


def test_sensitivity_strategy_name(early_strat):
    result = run_sensitivity(early_strat, n_seeds=5, num_agents=10, num_timesteps=5)
    assert result.strategy_name == early_strat.name


# ── CI ordering ───────────────────────────────────────────────────────────────

def test_ci_lower_le_mean_le_upper(early_strat):
    result = run_sensitivity(early_strat, n_seeds=10, num_agents=10, num_timesteps=8)
    for lo, mid, hi in zip(result.ci_lower_95, result.mean_belief_timeline, result.ci_upper_95):
        assert lo <= mid + 1e-9, f"lower {lo} > mean {mid}"
        assert mid <= hi + 1e-9, f"mean {mid} > upper {hi}"


# ── Value bounds ──────────────────────────────────────────────────────────────

def test_all_values_in_unit_interval(early_strat):
    result = run_sensitivity(early_strat, n_seeds=6, num_agents=10, num_timesteps=6)
    all_vals = (
        result.mean_belief_timeline
        + result.ci_lower_95
        + result.ci_upper_95
        + result.std_timeline
    )
    for v in all_vals:
        assert 0.0 <= v <= 1.0, f"Value {v} outside [0, 1]"


def test_final_alignment_stats_in_range(early_strat):
    result = run_sensitivity(early_strat, n_seeds=10, num_agents=10, num_timesteps=5)
    assert 0.0 <= result.final_alignment_mean <= 1.0
    assert result.final_alignment_std >= 0.0
    assert 0.0 <= result.final_alignment_min <= result.final_alignment_max <= 1.0


# ── Summary ───────────────────────────────────────────────────────────────────

def test_summary_keys(early_strat):
    result = run_sensitivity(early_strat, n_seeds=5, num_agents=10, num_timesteps=5)
    s = result.summary()
    for key in ("strategy", "n_seeds", "final_mean", "final_std", "final_min",
                "final_max", "ci_width_final"):
        assert key in s, f"Missing key in summary: {key}"


def test_summary_ci_width_non_negative(early_strat):
    result = run_sensitivity(early_strat, n_seeds=10, num_agents=10, num_timesteps=5)
    assert result.summary()["ci_width_final"] >= 0.0


# ── Multi-strategy comparison ─────────────────────────────────────────────────

def test_compare_sensitivity_length(all_strats):
    results = compare_sensitivity(all_strats, n_seeds=5, num_agents=10, num_timesteps=5)
    assert len(results) == len(all_strats)


def test_compare_sensitivity_unique_names(all_strats):
    results = compare_sensitivity(all_strats, n_seeds=5, num_agents=10, num_timesteps=5)
    names = [r.strategy_name for r in results]
    assert len(set(names)) == len(names), f"Duplicate strategy names: {names}"


def test_compare_sensitivity_name_order(all_strats):
    results = compare_sensitivity(all_strats, n_seeds=5, num_agents=10, num_timesteps=5)
    for strat, result in zip(all_strats, results):
        assert result.strategy_name == strat.name


# ── Robustness ────────────────────────────────────────────────────────────────

def test_min_seeds_raises():
    """n_seeds < 2 must raise ValueError."""
    strat = CommunicationStrategy.early_authoritative()
    with pytest.raises(ValueError, match="n_seeds must be >= 2"):
        run_sensitivity(strat, n_seeds=1, num_agents=10, num_timesteps=5)


def test_different_seeds_produce_variance(all_strats):
    """With 15+ seeds, std_timeline should be non-zero for at least some timesteps."""
    strat = all_strats[0]
    result = run_sensitivity(strat, n_seeds=15, num_agents=20, num_timesteps=10)
    # At least some timesteps should have non-zero variance across 15 seeds
    assert any(v > 0 for v in result.std_timeline), (
        "Expected non-zero variance across seeds"
    )


def test_early_strat_higher_mean_than_late():
    """Early strategy should have higher mean final alignment than late reactive."""
    strats = CommunicationStrategy.all_defaults()
    results = {
        r.strategy_name: r
        for r in compare_sensitivity(
            strats, n_seeds=20, num_agents=30, num_timesteps=15
        )
    }
    early = results.get("Early & Authoritative")
    late = results.get("Late & Reactive")
    if early and late:
        assert early.final_alignment_mean >= late.final_alignment_mean, (
            f"Early ({early.final_alignment_mean:.3f}) should >= "
            f"Late ({late.final_alignment_mean:.3f})"
        )
