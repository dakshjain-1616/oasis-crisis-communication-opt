"""
02_advanced_usage.py — Advanced features

Demonstrates:
- Custom StrategyConfig with explicit parameters
- BeliefTracker for per-agent belief tracking
- ResultsAnalyzer for effect sizes and time-to-threshold
- SimulationVisualizer for chart generation

Usage:
    python examples/02_advanced_usage.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("FORCE_MOCK_MODE", "1")

from oasis_crisis_communi.strategies import StrategyConfig, CommunicationStrategy
from oasis_crisis_communi.simulation import CrisisSimulation
from oasis_crisis_communi.belief_tracker import BeliefTracker
from oasis_crisis_communi.analyzer import ResultsAnalyzer
from oasis_crisis_communi.visualizer import SimulationVisualizer

# ── 1. Define custom strategies ────────────────────────────────────────────────
rapid_blitz = StrategyConfig(
    name="Rapid Blitz",
    start_timestep=1,
    tone_multiplier=1.0,       # authoritative
    posts_per_timestep=4,      # high frequency
    multi_channel=True,        # broadcast across platforms
    tone_label="authoritative",
    color="#E53935",
    description="Maximum early saturation across all channels",
)

slow_build = StrategyConfig(
    name="Slow Build",
    start_timestep=5,
    tone_multiplier=0.85,      # empathetic
    posts_per_timestep=2,
    multi_channel=False,
    tone_label="empathetic",
    color="#43A047",
    description="Gradual trust-building approach",
)

# Mix with a built-in strategy
late_reactive = CommunicationStrategy.late_reactive()

strategies = [rapid_blitz, slow_build, late_reactive]

# ── 2. Run simulation ──────────────────────────────────────────────────────────
sim = CrisisSimulation(num_agents=50, num_timesteps=25, num_misinfo_agents=8, seed=99)
results = sim.run_comparison(strategies)

# ── 3. Effect sizes and threshold timing ──────────────────────────────────────
analyzer = ResultsAnalyzer(results)
effect_sizes = analyzer.compute_effect_sizes()
thresholds = analyzer.compute_time_to_threshold(threshold=0.60)

print("=== Effect Sizes (Cohen's d) ===")
for pair, d in effect_sizes.items():
    print(f"  {pair}: d={d:.3f}")

print("\n=== Time to 60% Alignment Threshold ===")
for strat, t in thresholds.items():
    hit = f"timestep {t}" if t is not None else "never reached"
    print(f"  {strat}: {hit}")

# ── 4. BeliefTracker demo ──────────────────────────────────────────────────────
tracker = BeliefTracker()
sample_responses = [
    "I trust the official guidance. The evidence is credible.",
    "I'm uncertain — there's too much conflicting information.",
    "I don't believe a word of it. This is propaganda.",
]
for i, text in enumerate(sample_responses):
    resp = tracker.parse_interview_response(agent_id=i, timestep=10, raw_response=text)
    print(f"\nAgent {i}: belief={resp.belief_score:.2f}, sentiment={resp.sentiment}")

print(f"\nPopulation alignment at t=10: {tracker.get_population_alignment(10):.1%}")

# ── 5. Generate a chart ────────────────────────────────────────────────────────
output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(output_dir, exist_ok=True)
viz = SimulationVisualizer(results, output_dir=output_dir)
chart = viz.plot_belief_alignment()
print(f"\nChart saved: {chart}")
