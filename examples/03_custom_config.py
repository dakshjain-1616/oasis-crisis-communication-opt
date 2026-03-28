"""
03_custom_config.py — Customising behaviour via environment variables / config

Demonstrates how every simulation parameter can be controlled via env vars
or passed directly, and how to use pre-built scenario cards.

Usage:
    # Default run (reads from environment)
    python examples/03_custom_config.py

    # Override via env vars
    NUM_AGENTS=80 NUM_TIMESTEPS=30 RANDOM_SEED=7 python examples/03_custom_config.py

    # Use a specific scenario card
    SCENARIO=bioterrorism python examples/03_custom_config.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("FORCE_MOCK_MODE", "1")

from oasis_crisis_communi.simulation import CrisisSimulation
from oasis_crisis_communi.analyzer import ResultsAnalyzer
from oasis_crisis_communi.scenarios import list_scenarios, get_scenario
from oasis_crisis_communi.reporter import ReportGenerator

# ── Read config from environment (with defaults) ───────────────────────────────
NUM_AGENTS     = int(os.getenv("NUM_AGENTS", "40"))
NUM_TIMESTEPS  = int(os.getenv("NUM_TIMESTEPS", "20"))
NUM_MISINFO    = int(os.getenv("NUM_MISINFO_AGENTS", "5"))
SEED           = int(os.getenv("RANDOM_SEED", "42"))
SCENARIO_ID    = os.getenv("SCENARIO", "")           # e.g. "covid_pandemic"
OUTPUT_DIR     = os.getenv("OUTPUT_DIR", "outputs")

print("=== Configuration ===")
print(f"  Agents:    {NUM_AGENTS}")
print(f"  Timesteps: {NUM_TIMESTEPS}")
print(f"  Misinfo:   {NUM_MISINFO}")
print(f"  Seed:      {SEED}")
print(f"  Scenario:  {SCENARIO_ID or 'default (3 built-in strategies)'}")
print()

# ── Pick strategies ────────────────────────────────────────────────────────────
if SCENARIO_ID:
    try:
        scenario = get_scenario(SCENARIO_ID)
        strategies = scenario.recommended_strategies
        # Override sim params from scenario if not explicitly set
        if "NUM_AGENTS" not in os.environ:
            NUM_AGENTS = scenario.sim_params.get("n_citizens", NUM_AGENTS)
        if "NUM_TIMESTEPS" not in os.environ:
            NUM_TIMESTEPS = scenario.sim_params.get("n_timesteps", NUM_TIMESTEPS)
        if "NUM_MISINFO_AGENTS" not in os.environ:
            NUM_MISINFO = scenario.sim_params.get("n_misinfo_agents", NUM_MISINFO)
        print(f"Loaded scenario: {scenario.name} [{scenario.severity.upper()}]")
        print(f"  {scenario.description}")
    except KeyError:
        valid = [s.id for s in list_scenarios()]
        print(f"Unknown SCENARIO={SCENARIO_ID!r}. Valid options: {valid}")
        sys.exit(1)
else:
    from oasis_crisis_communi.strategies import CommunicationStrategy
    strategies = CommunicationStrategy.all_defaults()

print(f"\nStrategies: {[s.name for s in strategies]}\n")

# ── Run simulation ─────────────────────────────────────────────────────────────
sim = CrisisSimulation(
    num_agents=NUM_AGENTS,
    num_timesteps=NUM_TIMESTEPS,
    num_misinfo_agents=NUM_MISINFO,
    seed=SEED,
)
results = sim.run_comparison(strategies)

# ── Print summary table ────────────────────────────────────────────────────────
analyzer = ResultsAnalyzer(results)
summary = analyzer.get_summary_table()
print("=== Results ===")
print(summary[["strategy", "final_alignment", "peak_alignment", "avg_alignment"]].to_string(index=False))

# ── Save JSON report ───────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
reporter = ReportGenerator(results, output_dir=OUTPUT_DIR)
json_path = reporter.save_json_report()
print(f"\nJSON report: {json_path}")

# ── List all available scenarios ───────────────────────────────────────────────
print("\n=== Available Scenario IDs ===")
for s in list_scenarios():
    print(f"  {s.id:25s} — {s.name} ({s.severity})")
print("\nTip: set SCENARIO=<id> to run a pre-built scenario.")
