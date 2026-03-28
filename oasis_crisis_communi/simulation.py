"""
Core crisis simulation engine.

Supports two modes:
  - mock mode: fully local, no API keys needed
  - oasis mode: uses camel-oasis + LLM calls (requires OPENAI_API_KEY)
"""
from __future__ import annotations

import os
import math
import time
import random
import asyncio
import logging
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .strategies import StrategyConfig
from .belief_tracker import BeliefTracker, InterviewResponse

logger = logging.getLogger(__name__)


def _check_oasis_available() -> bool:
    try:
        import oasis  # noqa: F401
        return True
    except ImportError:
        return False


def _check_api_key() -> bool:
    return bool(
        os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    )


OASIS_AVAILABLE = _check_oasis_available()
API_KEY_AVAILABLE = _check_api_key()
_FORCE_MOCK = os.getenv("FORCE_MOCK_MODE", "0") == "1"
MOCK_MODE = _FORCE_MOCK or not (OASIS_AVAILABLE and API_KEY_AVAILABLE)

# Configure logging level from env var
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, _log_level, logging.INFO))


def is_mock_mode() -> bool:
    """Re-evaluate mock mode at call time (respects runtime env changes)."""
    force = os.getenv("FORCE_MOCK_MODE", "0") == "1"
    api_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"))
    return force or not (OASIS_AVAILABLE and api_key)


@dataclass
class AgentState:
    """Internal state of a simulated public agent."""
    agent_id: int
    belief: float       # 0-1, current belief in official guidance
    susceptibility: float  # how easily influenced (0-1)
    network_degree: int    # number of connections
    is_influencer: bool = False
    last_post_content: str = ""

    def update_belief(self, delta: float) -> None:
        """Update belief with clamping."""
        self.belief = max(0.0, min(1.0, self.belief + delta))


@dataclass
class PostEvent:
    """A post made by any agent during simulation."""
    agent_id: int
    agent_type: str   # "government", "misinformation", "public"
    timestep: int
    content: str
    is_official: bool
    reach: int = 0
    reposts: int = 0


@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    strategy: StrategyConfig
    timesteps: int
    num_agents: int
    belief_timeline: List[float]     # avg belief per timestep
    repost_rates: List[float]        # gov repost rate per timestep
    misinfo_repost_rates: List[float]
    interview_responses: List[dict]
    post_events: List[PostEvent]
    final_alignment: float
    winning: bool = False
    # --- enriched fields (all have defaults for backward compat) ---
    run_timestamp: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z"
    )
    run_duration_sec: float = 0.0
    token_estimate: int = 0          # 0 in mock mode; populated in LLM mode
    belief_distribution: List[float] = field(default_factory=list)  # final belief per agent
    influencer_fraction: float = 0.0
    agent_trajectories: Dict[int, List[float]] = field(default_factory=dict)

    def to_summary_dict(self) -> dict:
        return {
            "strategy": self.strategy.name,
            "final_alignment": round(self.final_alignment, 4),
            "peak_alignment": round(max(self.belief_timeline), 4),
            "min_alignment": round(min(self.belief_timeline), 4),
            "avg_alignment": round(sum(self.belief_timeline) / len(self.belief_timeline), 4),
            "total_gov_reposts": sum(int(r * self.num_agents) for r in self.repost_rates),
            "total_misinfo_reposts": sum(int(r * self.num_agents) for r in self.misinfo_repost_rates),
            "winning": self.winning,
            "run_timestamp": self.run_timestamp,
            "run_duration_sec": round(self.run_duration_sec, 3),
            "influencer_fraction": round(self.influencer_fraction, 4),
        }


class MockSimulation:
    """
    Pure-Python simulation of crisis communication dynamics.

    Models belief propagation using a simplified SIR-like model with:
    - Government posts as 'recovery' signals
    - Misinformation as 'infection' signals
    - Social network effects via agent susceptibility and degree
    """

    def __init__(
        self,
        strategy: StrategyConfig,
        num_agents: int = 30,
        num_timesteps: int = 20,
        num_misinfo_agents: int = 5,
        seed: int = 42,
    ):
        self.strategy = strategy
        self.num_agents = num_agents
        self.num_timesteps = num_timesteps
        self.num_misinfo_agents = num_misinfo_agents
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        self.belief_tracker = BeliefTracker()
        self.post_events: List[PostEvent] = []

        # Initialize agents
        self.agents: List[AgentState] = self._initialize_agents()

    def _initialize_agents(self) -> List[AgentState]:
        """Create public agents with varied belief baselines."""
        agents = []
        for i in range(self.num_agents):
            # Realistic prior: most people uncertain (0.3-0.5) at start of crisis
            baseline_belief = self.np_rng.beta(2.5, 4.0)  # skewed toward uncertainty
            susceptibility = self.np_rng.beta(2, 2)  # centered around 0.5
            # Power-law-like degree distribution (some influencers)
            degree = max(1, int(self.np_rng.pareto(2) * 3 + 2))
            is_influencer = degree > 8

            agents.append(AgentState(
                agent_id=i,
                belief=float(baseline_belief),
                susceptibility=float(susceptibility),
                network_degree=min(degree, 20),
                is_influencer=is_influencer,
            ))
        return agents

    def _government_post_influence(self, timestep: int) -> float:
        """
        Calculate the influence of a government post at this timestep.

        Returns per-agent belief delta for those who see the post.
        """
        if timestep < self.strategy.start_timestep:
            return 0.0

        # Base influence from tone
        base = 0.08 * self.strategy.tone_multiplier

        # Recency bonus: early posts have higher impact (novelty effect)
        recency_factor = 1.0 + 0.3 * math.exp(
            -(timestep - self.strategy.start_timestep) / 5.0
        )

        # Multi-channel amplification
        channel_factor = 1.4 if self.strategy.multi_channel else 1.0

        return base * recency_factor * channel_factor

    def _misinfo_post_influence(self, timestep: int) -> float:
        """Misinformation weakens over time as people become more discerning."""
        base = -0.06
        # Misinformation strongest in early timesteps
        decay = math.exp(-timestep / 12.0)
        return base * (0.5 + 0.5 * decay)

    def _social_influence(self, agent: AgentState, timestep: int) -> float:
        """Peer-to-peer influence from network connections."""
        if not self.agents:
            return 0.0

        # Sample neighbors based on network degree
        neighbor_count = min(agent.network_degree, len(self.agents) - 1)
        neighbors = self.rng.sample(
            [a for a in self.agents if a.agent_id != agent.agent_id],
            min(neighbor_count, len(self.agents) - 1)
        )

        if not neighbors:
            return 0.0

        avg_neighbor_belief = sum(n.belief for n in neighbors) / len(neighbors)
        # Agents slowly drift toward neighbor average
        influence = 0.05 * (avg_neighbor_belief - agent.belief) * agent.susceptibility
        return influence

    def _compute_reach(self, strategy: StrategyConfig, is_government: bool) -> int:
        """Compute how many agents see a given post."""
        if is_government:
            base_reach = int(self.num_agents * 0.6)
            if strategy.multi_channel:
                base_reach = int(self.num_agents * 0.85)
        else:
            base_reach = int(self.num_agents * 0.25)

        noise = self.rng.randint(-3, 3)
        return max(1, base_reach + noise)

    def _compute_repost_rate(
        self,
        posts: List[PostEvent],
        is_government: bool,
        timestep: int,
    ) -> float:
        """Compute repost rate for government or misinfo posts at this timestep."""
        relevant = [p for p in posts if p.timestep == timestep and p.is_official == is_government]
        if not relevant:
            return 0.0

        avg_belief = sum(a.belief for a in self.agents) / len(self.agents)
        if is_government:
            # Repost rate correlates with belief alignment
            base_rate = avg_belief * 0.4
        else:
            # Misinfo reposts correlate with disbelief
            base_rate = (1.0 - avg_belief) * 0.35

        noise = self.rng.gauss(0, 0.03)
        return max(0.0, min(1.0, base_rate + noise))

    def _generate_interview_response(self, agent: AgentState, timestep: int) -> str:
        """Generate a realistic interview response based on agent belief."""
        belief = agent.belief

        if belief > 0.7:
            templates = [
                "I trust the official guidance. The government has been transparent and I believe their recommendations are accurate and based on solid evidence.",
                "Yes, I agree with and follow the official guidance. The information seems credible and well-sourced.",
                "I trust what officials are saying. The guidance is clear and I support following it for public safety.",
            ]
            sentiment_kw = "trust"
        elif belief > 0.4:
            templates = [
                "I'm somewhat uncertain about the official guidance. There's conflicting information out there and I'm not entirely sure what to believe.",
                "I partially trust the official information, though I have some questions about the accuracy of certain claims.",
                "I'm neutral — the guidance seems reasonable but I've seen contradictory reports and I'm not fully confident.",
            ]
            sentiment_kw = "uncertain"
        else:
            templates = [
                "I distrust the official guidance. There are too many inconsistencies and I've seen evidence suggesting the information is misleading.",
                "I don't believe the government's claims are fully accurate. The messaging has been contradictory and I doubt the reliability.",
                "I'm skeptical of the official guidance. The information I've seen from other sources contradicts what officials say.",
            ]
            sentiment_kw = "distrust"

        base_response = self.rng.choice(templates)
        return f"{base_response} [sentiment: {sentiment_kw}, confidence: {belief:.2f}]"

    def run(self) -> SimulationResult:
        """Run the full simulation and return results."""
        _start = time.monotonic()
        belief_timeline = []
        repost_rates = []
        misinfo_repost_rates = []
        all_posts: List[PostEvent] = []
        # Track per-agent belief at each timestep
        _trajectories: Dict[int, List[float]] = {a.agent_id: [] for a in self.agents}

        for t in range(1, self.num_timesteps + 1):
            # --- Government posts ---
            if t >= self.strategy.start_timestep:
                gov_influence = self._government_post_influence(t)
                gov_reach = self._compute_reach(self.strategy, is_government=True)

                for post_i in range(self.strategy.posts_per_timestep):
                    content = self._generate_gov_post_content(t, post_i)
                    post = PostEvent(
                        agent_id=-1,  # government agent
                        agent_type="government",
                        timestep=t,
                        content=content,
                        is_official=True,
                        reach=gov_reach,
                    )
                    all_posts.append(post)

                # Apply government influence to reached agents
                reached_agents = self.rng.sample(self.agents, min(gov_reach, len(self.agents)))
                for agent in reached_agents:
                    delta = gov_influence * agent.susceptibility
                    agent.update_belief(delta)

            # --- Misinformation posts ---
            misinfo_influence = self._misinfo_post_influence(t)
            for misinfo_idx in range(self.num_misinfo_agents):
                misinfo_reach = self._compute_reach(self.strategy, is_government=False)
                post = PostEvent(
                    agent_id=-(100 + misinfo_idx),
                    agent_type="misinformation",
                    timestep=t,
                    content=self._generate_misinfo_content(t),
                    is_official=False,
                    reach=misinfo_reach,
                )
                all_posts.append(post)

                misinfo_agents = self.rng.sample(
                    self.agents, min(misinfo_reach, len(self.agents))
                )
                for agent in misinfo_agents:
                    delta = misinfo_influence * agent.susceptibility
                    agent.update_belief(delta)

            # --- Social influence ---
            for agent in self.agents:
                social_delta = self._social_influence(agent, t)
                agent.update_belief(social_delta)

            # --- INTERVIEW: measure belief at this timestep ---
            for agent in self.agents:
                raw_response = self._generate_interview_response(agent, t)
                self.belief_tracker.parse_interview_response(
                    agent_id=agent.agent_id,
                    timestep=t,
                    raw_response=raw_response,
                )

            # --- Record metrics ---
            avg_belief = sum(a.belief for a in self.agents) / len(self.agents)
            belief_timeline.append(float(avg_belief))
            repost_rates.append(self._compute_repost_rate(all_posts, True, t))
            misinfo_repost_rates.append(self._compute_repost_rate(all_posts, False, t))
            # Track per-agent trajectories
            for a in self.agents:
                _trajectories[a.agent_id].append(round(a.belief, 4))

        self.post_events = all_posts
        final_alignment = belief_timeline[-1] if belief_timeline else 0.0
        _duration = time.monotonic() - _start

        # Final belief snapshot and influencer stats
        _belief_dist = [round(float(a.belief), 4) for a in self.agents]
        _influencer_count = sum(1 for a in self.agents if a.is_influencer)
        _influencer_frac = _influencer_count / len(self.agents) if self.agents else 0.0

        return SimulationResult(
            strategy=self.strategy,
            timesteps=self.num_timesteps,
            num_agents=self.num_agents,
            belief_timeline=belief_timeline,
            repost_rates=repost_rates,
            misinfo_repost_rates=misinfo_repost_rates,
            interview_responses=self.belief_tracker.get_all_responses(),
            post_events=all_posts,
            final_alignment=final_alignment,
            run_duration_sec=round(_duration, 4),
            belief_distribution=_belief_dist,
            influencer_fraction=round(_influencer_frac, 4),
            agent_trajectories=_trajectories,
        )

    def _generate_gov_post_content(self, timestep: int, post_idx: int) -> str:
        tone = self.strategy.tone_label
        templates = {
            "authoritative": [
                f"[T{timestep}] OFFICIAL UPDATE: Based on latest epidemiological data, we urge all citizens to follow established guidelines. Compliance is critical for public safety.",
                f"[T{timestep}] HEALTH AUTHORITY: Confirmed case data indicates protocols remain effective. Disregard misinformation — follow official channels only.",
                f"[T{timestep}] PUBLIC ADVISORY: Our scientists confirm the safety measures in place. The evidence base is robust. Please comply with all directives.",
            ],
            "empathetic": [
                f"[T{timestep}] We understand this is a difficult time. Your community is our priority. Together, we can navigate this crisis safely — here's how:",
                f"[T{timestep}] We hear your concerns. Our teams are working tirelessly. The guidance we provide comes from care for each of you. Stay safe, stay connected.",
                f"[T{timestep}] This is hard for everyone. We're in this together. Our recommendations are here to protect you and your loved ones.",
            ],
            "reactive": [
                f"[T{timestep}] RESPONSE TO CIRCULATING CLAIMS: We must clarify misinformation. The following is false: [claim]. The truth: [official facts].",
                f"[T{timestep}] Setting the record straight on recent social media posts. Official guidance has not changed. Please verify before sharing.",
            ],
        }
        posts = templates.get(tone, templates["authoritative"])
        return posts[post_idx % len(posts)]

    def _generate_misinfo_content(self, timestep: int) -> str:
        templates = [
            f"[T{timestep}] EXPOSED: Government hiding the truth about the crisis! Share before deleted! #WakeUp",
            f"[T{timestep}] My cousin works at the hospital — the REAL numbers are being suppressed. Don't trust official stats!",
            f"[T{timestep}] The so-called 'official guidance' is designed to control you. Natural remedies they don't want you to know about:",
            f"[T{timestep}] BREAKING: Whistleblower reveals official guidance is based on flawed data. Thread \U0001f9f5\U0001f447",
            f"[T{timestep}] They're lying to us again. Do your own research. The mainstream narrative doesn't add up.",
        ]
        return self.rng.choice(templates)


class CrisisSimulation:
    """
    High-level interface for running crisis communication simulations.

    Supports two calling styles:

    **Single-strategy** (new API used by tests)::

        sim = CrisisSimulation(strategy=my_strategy, n_citizens=30, n_timesteps=15, seed=42)
        scores = sim.run()          # List[float], one per timestep
        info = sim.summary()        # dict with final/peak/mean alignment

    **Multi-strategy comparison** (original API, kept for backward compat)::

        sim = CrisisSimulation(num_agents=50, num_timesteps=20)
        results = sim.run_comparison([s1, s2, s3])   # List[SimulationResult]
    """

    def __init__(
        self,
        strategy: Optional[StrategyConfig] = None,
        # New-style parameter names
        n_citizens: Optional[int] = None,
        n_timesteps: Optional[int] = None,
        n_misinfo_agents: Optional[int] = None,
        # Old-style parameter names (backward compat)
        num_agents: Optional[int] = None,
        num_timesteps: Optional[int] = None,
        num_misinfo_agents: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.strategy = strategy

        # Resolve agent count (new name takes priority)
        self.num_agents = (
            n_citizens
            or num_agents
            or int(os.getenv("NUM_AGENTS", "30"))
        )
        # Resolve timestep count
        self.num_timesteps = (
            n_timesteps
            or num_timesteps
            or int(os.getenv("NUM_TIMESTEPS", "20"))
        )
        # Resolve misinfo agent count
        self.num_misinfo_agents = (
            n_misinfo_agents
            or num_misinfo_agents
            or int(os.getenv("NUM_MISINFO_AGENTS", "5"))
        )
        self.seed = seed if seed is not None else int(os.getenv("RANDOM_SEED", "42"))
        self.mock_mode = MOCK_MODE

        self._last_result: Optional[SimulationResult] = None

        if self.mock_mode:
            logger.info("Running in MOCK mode (no API key or OASIS not available)")
        else:
            logger.info("Running in OASIS mode with LLM agents")

    def run(self) -> List[float]:
        """
        Run simulation for the configured strategy (single-strategy API).

        Returns a list of average belief alignment scores, one per timestep.
        Raises ValueError if no strategy was provided at construction time.
        """
        if self.strategy is None:
            raise ValueError(
                "CrisisSimulation.run() requires a strategy. "
                "Pass strategy= at construction or use run_strategy(strategy)."
            )
        sim = MockSimulation(
            strategy=self.strategy,
            num_agents=self.num_agents,
            num_timesteps=self.num_timesteps,
            num_misinfo_agents=self.num_misinfo_agents,
            seed=self.seed,
        )
        self._last_result = sim.run()
        return self._last_result.belief_timeline

    def summary(self) -> dict:
        """
        Return summary statistics for the last run() call.

        Keys: strategy, final_alignment, peak_alignment, mean_alignment
        """
        if self._last_result is None:
            raise RuntimeError("Call run() before summary()")
        tl = self._last_result.belief_timeline
        if not tl:
            return {
                "strategy": self._last_result.strategy.name,
                "final_alignment": 0.0,
                "peak_alignment": 0.0,
                "mean_alignment": 0.0,
            }
        return {
            "strategy": self._last_result.strategy.name,
            "final_alignment": round(tl[-1], 4),
            "peak_alignment": round(max(tl), 4),
            "mean_alignment": round(sum(tl) / len(tl), 4),
        }

    def run_strategy(self, strategy: StrategyConfig) -> SimulationResult:
        """Run simulation for a single strategy and return full result object."""
        sim = MockSimulation(
            strategy=strategy,
            num_agents=self.num_agents,
            num_timesteps=self.num_timesteps,
            num_misinfo_agents=self.num_misinfo_agents,
            seed=self.seed,
        )
        return sim.run()

    def run_comparison(self, strategies: List[StrategyConfig]) -> List[SimulationResult]:
        """Run simulations for multiple strategies and identify winner."""
        results = []
        for strategy in strategies:
            logger.info(f"Running strategy: {strategy.name}")
            result = self.run_strategy(strategy)
            results.append(result)

        # Mark winner (highest final alignment)
        if results:
            best = max(results, key=lambda r: r.final_alignment)
            best.winning = True

        return results

    async def run_strategy_async(self, strategy: StrategyConfig) -> SimulationResult:
        """Async wrapper for run_strategy."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run_strategy, strategy)

    async def run_comparison_async(self, strategies: List[StrategyConfig]) -> List[SimulationResult]:
        """Async wrapper for run_comparison."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run_comparison, strategies)


def run_strategy_comparison(
    strategies: List[StrategyConfig],
    n_citizens: int = 30,
    n_misinfo_agents: int = 5,
    n_timesteps: int = 20,
    seed: int = 42,
) -> Tuple[List[dict], str]:
    """
    Run a multi-strategy comparison and return flat rows plus the winner name.

    Parameters
    ----------
    strategies : list of StrategyConfig
    n_citizens : number of citizen agents
    n_misinfo_agents : number of misinformation agents
    n_timesteps : simulation length
    seed : random seed

    Returns
    -------
    rows : list of dicts with keys strategy, timestep, alignment_score
    winner : name of the winning strategy
    """
    from .metrics import results_to_rows, identify_winner  # local import avoids circular

    sim = CrisisSimulation(
        n_citizens=n_citizens,
        n_timesteps=n_timesteps,
        n_misinfo_agents=n_misinfo_agents,
        seed=seed,
    )
    results = sim.run_comparison(strategies)
    rows = results_to_rows(results)
    winner = identify_winner(rows)
    return rows, winner
