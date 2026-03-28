"""
01_quick_start.py — Minimal working example

Runs a single default strategy comparison with 3 strategies, 30 agents,
20 timesteps in mock mode (no API key needed). Prints the winner.

Usage:
    python examples/01_quick_start.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("FORCE_MOCK_MODE", "1")

from oasis_crisis_communi.strategies import CommunicationStrategy
from oasis_crisis_communi.simulation import CrisisSimulation
from oasis_crisis_communi.analyzer import ResultsAnalyzer

# 1. Get the three built-in strategies
strategies = CommunicationStrategy.all_defaults()

# 2. Run the comparison (30 agents, 20 timesteps, 5 misinfo agents)
sim = CrisisSimulation(num_agents=30, num_timesteps=20, num_misinfo_agents=5, seed=42)
results = sim.run_comparison(strategies)

# 3. Analyse and print the winner
analyzer = ResultsAnalyzer(results)
winner = analyzer.get_winner()
print(f"Winner: {winner.strategy.name} — {winner.final_alignment:.1%} alignment")
print(analyzer.generate_recommendation())
