"""
04_full_pipeline.py — End-to-end workflow

Demonstrates the complete project pipeline:
1. Load a pre-built scenario (COVID pandemic)
2. Run multi-strategy comparison simulation
3. Run sensitivity analysis (multi-seed robustness)
4. Analyse results: effect sizes, threshold timing, belief distribution
5. Generate all output artefacts: CSV, PNG charts, JSON + HTML reports
6. Print a plain-English recommendation

Usage:
    python examples/04_full_pipeline.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("FORCE_MOCK_MODE", "1")

import json

from oasis_crisis_communi.scenarios import get_scenario
from oasis_crisis_communi.simulation import CrisisSimulation
from oasis_crisis_communi.analyzer import ResultsAnalyzer
from oasis_crisis_communi.visualizer import SimulationVisualizer
from oasis_crisis_communi.reporter import ReportGenerator
from oasis_crisis_communi.sensitivity import compare_sensitivity
from oasis_crisis_communi.metrics import results_to_rows, save_alignment_csv

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("OASIS Crisis Communication Optimizer — Full Pipeline")
print("=" * 60)

# ── Step 1: Load scenario ──────────────────────────────────────────────────────
scenario = get_scenario("covid_pandemic")
print(f"\n[1] Scenario: {scenario.name} ({scenario.severity.upper()})")
print(f"    {scenario.description}")
strategies = scenario.recommended_strategies
sim_params = scenario.sim_params
print(f"    Strategies: {[s.name for s in strategies]}")
print(f"    Params: {sim_params['n_citizens']} agents, "
      f"{sim_params['n_timesteps']} timesteps, "
      f"{sim_params['n_misinfo_agents']} misinfo agents")

# ── Step 2: Multi-strategy simulation ─────────────────────────────────────────
print("\n[2] Running simulation...")
sim = CrisisSimulation(
    num_agents=sim_params["n_citizens"],
    num_timesteps=sim_params["n_timesteps"],
    num_misinfo_agents=sim_params["n_misinfo_agents"],
    seed=42,
)
results = sim.run_comparison(strategies)

analyzer = ResultsAnalyzer(results)
summary = analyzer.get_summary_table()
winner = analyzer.get_winner()
print(f"    Winner: {winner.strategy.name} — {winner.final_alignment:.1%} alignment")

# ── Step 3: Sensitivity analysis (robustness across seeds) ────────────────────
print("\n[3] Running sensitivity analysis (10 seeds)...")
sen_results = compare_sensitivity(
    strategies,
    n_seeds=10,
    num_agents=sim_params["n_citizens"],
    num_timesteps=sim_params["n_timesteps"],
)
most_robust = min(sen_results, key=lambda r: r.final_alignment_std)
best_mean   = max(sen_results, key=lambda r: r.final_alignment_mean)
print(f"    Most robust (lowest σ): {most_robust.strategy_name} "
      f"(σ={most_robust.final_alignment_std:.3f})")
print(f"    Best mean performance:  {best_mean.strategy_name} "
      f"(μ={best_mean.final_alignment_mean:.1%})")

# ── Step 4: Analysis ──────────────────────────────────────────────────────────
print("\n[4] Analysis:")
effect_sizes = analyzer.compute_effect_sizes()
thresholds   = analyzer.compute_time_to_threshold(threshold=0.60)
influencers  = analyzer.compute_influencer_impact()

print("    Effect sizes (Cohen's d):")
for pair, d in effect_sizes.items():
    print(f"      {pair}: d={d:.3f}")

print("    Time to 60% threshold:")
for strat, t in thresholds.items():
    print(f"      {strat}: {'t=' + str(t) if t else 'not reached'}")

if winner:
    inf = influencers.get(winner.strategy.name, {})
    print(f"    Winning strategy belief distribution: "
          f"μ={inf.get('belief_mean', 0):.2f}, "
          f"σ={inf.get('belief_std', 0):.2f}, "
          f"{inf.get('pct_above_60', 0):.0%} above 60%")

# ── Step 5: Generate all output artefacts ─────────────────────────────────────
print("\n[5] Saving outputs...")

# CSV
rows = results_to_rows(results)
csv_path = save_alignment_csv(rows, output_path=os.path.join(OUTPUT_DIR, "belief_alignment.csv"))
print(f"    CSV:          {csv_path}")

# Charts
viz = SimulationVisualizer(results, output_dir=OUTPUT_DIR)
comparison_chart = viz.plot_belief_alignment()
print(f"    Comparison:   {comparison_chart}")

if winner:
    dist_chart = viz.plot_belief_distribution(winner)
    print(f"    Distribution: {dist_chart}")

# JSON + HTML reports
reporter = ReportGenerator(results, output_dir=OUTPUT_DIR)
json_path = reporter.save_json_report()
html_path = reporter.save_html_report()
print(f"    JSON report:  {json_path}")
print(f"    HTML report:  {html_path}")

# Interview samples
samples = analyzer.get_interview_sample(5)
samples_path = os.path.join(OUTPUT_DIR, "interview_samples.json")
with open(samples_path, "w") as f:
    json.dump(samples, f, indent=2)
print(f"    Interviews:   {samples_path}")

# ── Step 6: Recommendation ────────────────────────────────────────────────────
print("\n[6] Recommendation:")
print("-" * 60)
print(analyzer.generate_recommendation())
print("-" * 60)
print("\nFull pipeline complete.")
