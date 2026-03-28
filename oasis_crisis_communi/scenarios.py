"""
Crisis Scenario Cards for the OASIS Crisis Communication Optimizer.

Each ScenarioCard bundles a realistic public-health emergency scenario with
recommended strategy configurations and simulation parameters so users can
jump straight to a meaningful comparison without manual tuning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .strategies import StrategyConfig


@dataclass
class ScenarioCard:
    """A pre-built crisis scenario with recommended communication strategies."""
    id: str                                      # machine-readable slug
    name: str                                    # display name
    description: str                             # 2-3 sentence summary
    crisis_type: str                             # pandemic | outbreak | natural_disaster | bioterrorism
    recommended_strategies: List[StrategyConfig]
    sim_params: Dict                             # n_citizens, n_timesteps, n_misinfo_agents
    icon: str = ""
    severity: str = "high"                       # low | medium | high | critical


# ── Scenario 1: COVID-style pandemic ─────────────────────────────────────────

SCENARIO_COVID_PANDEMIC = ScenarioCard(
    id="covid_pandemic",
    name="COVID-19 Style Pandemic",
    icon="🦠",
    severity="critical",
    crisis_type="pandemic",
    description=(
        "A rapidly spreading respiratory pandemic with high public anxiety and "
        "heavy misinformation on social media. Early, empathetic multi-channel "
        "communication is critical to maintaining public trust over many weeks."
    ),
    sim_params={"n_citizens": 80, "n_timesteps": 30, "n_misinfo_agents": 8},
    recommended_strategies=[
        StrategyConfig(
            name="Early Empathetic Blitz",
            start_timestep=1,
            tone_multiplier=0.85,
            posts_per_timestep=3,
            multi_channel=True,
            tone_label="empathetic",
            color="#4CAF50",
            description=(
                "Immediate, high-frequency empathetic messaging across all channels. "
                "Addresses fear and uncertainty head-on from day one."
            ),
        ),
        StrategyConfig(
            name="Authoritative Early",
            start_timestep=1,
            tone_multiplier=1.0,
            posts_per_timestep=2,
            multi_channel=False,
            tone_label="authoritative",
            color="#2196F3",
            description=(
                "Official fact-based guidance from the start. Moderate frequency, "
                "single-channel. Relies on institutional credibility."
            ),
        ),
        StrategyConfig(
            name="Delayed Crisis Response",
            start_timestep=10,
            tone_multiplier=0.95,
            posts_per_timestep=2,
            multi_channel=True,
            tone_label="reactive",
            color="#F44336",
            description=(
                "Government waits 10 timesteps before responding, then ramps up "
                "reactive messaging to counter established misinformation."
            ),
        ),
    ],
)


# ── Scenario 2: Infectious disease outbreak ───────────────────────────────────

SCENARIO_DISEASE_OUTBREAK = ScenarioCard(
    id="disease_outbreak",
    name="Infectious Disease Outbreak",
    icon="🏥",
    severity="high",
    crisis_type="outbreak",
    description=(
        "A localised but fast-moving disease outbreak (e.g. Ebola, Mpox) where "
        "clear authoritative guidance is needed quickly. Misinformation spreads "
        "at moderate pace; early action dramatically improves outcomes."
    ),
    sim_params={"n_citizens": 50, "n_timesteps": 20, "n_misinfo_agents": 5},
    recommended_strategies=[
        StrategyConfig(
            name="Rapid Authoritative",
            start_timestep=1,
            tone_multiplier=1.0,
            posts_per_timestep=2,
            multi_channel=False,
            tone_label="authoritative",
            color="#2196F3",
            description=(
                "Immediate authoritative guidance from health authorities. "
                "Evidence-based, medium frequency."
            ),
        ),
        StrategyConfig(
            name="Community Empathy",
            start_timestep=1,
            tone_multiplier=0.85,
            posts_per_timestep=3,
            multi_channel=True,
            tone_label="empathetic",
            color="#4CAF50",
            description=(
                "Community-focused empathetic messaging to build trust and reduce "
                "panic while delivering accurate guidance."
            ),
        ),
        StrategyConfig(
            name="Watch and React",
            start_timestep=6,
            tone_multiplier=0.95,
            posts_per_timestep=1,
            multi_channel=False,
            tone_label="reactive",
            color="#FF9800",
            description=(
                "Conservative approach — waits until outbreak is declared before "
                "issuing reactive single-channel corrections."
            ),
        ),
    ],
)


# ── Scenario 3: Natural disaster ─────────────────────────────────────────────

SCENARIO_NATURAL_DISASTER = ScenarioCard(
    id="natural_disaster",
    name="Hurricane / Natural Disaster",
    icon="🌀",
    severity="high",
    crisis_type="natural_disaster",
    description=(
        "A major hurricane or natural disaster requiring immediate evacuation and "
        "safety guidance. Misinformation about routes and shelter spreads quickly "
        "on social media. Speed and clarity are paramount."
    ),
    sim_params={"n_citizens": 60, "n_timesteps": 15, "n_misinfo_agents": 6},
    recommended_strategies=[
        StrategyConfig(
            name="Rapid Broadcast",
            start_timestep=1,
            tone_multiplier=1.0,
            posts_per_timestep=4,
            multi_channel=True,
            tone_label="authoritative",
            color="#2196F3",
            description=(
                "Maximum frequency authoritative broadcast from timestep 1 across "
                "all channels. Saturation strategy for urgent safety information."
            ),
        ),
        StrategyConfig(
            name="Calm & Clear",
            start_timestep=1,
            tone_multiplier=0.85,
            posts_per_timestep=2,
            multi_channel=True,
            tone_label="empathetic",
            color="#4CAF50",
            description=(
                "Empathetic tone to prevent panic while delivering clear "
                "evacuation instructions. Multi-channel reach."
            ),
        ),
        StrategyConfig(
            name="Reactive Corrections",
            start_timestep=4,
            tone_multiplier=0.95,
            posts_per_timestep=2,
            multi_channel=False,
            tone_label="reactive",
            color="#F44336",
            description=(
                "Waits for misinformation to establish itself then issues targeted "
                "corrections. Risky given the short response window."
            ),
        ),
    ],
)


# ── Scenario 4: Bioterrorism / deliberate contamination ──────────────────────

SCENARIO_BIOTERRORISM = ScenarioCard(
    id="bioterrorism",
    name="Bioterrorism / Deliberate Contamination",
    icon="⚠️",
    severity="critical",
    crisis_type="bioterrorism",
    description=(
        "A confirmed or suspected deliberate biological attack or contamination "
        "event. Public trust is fragile; conspiracy theories spread extremely fast. "
        "Transparent, authoritative communication is critical to prevent mass panic."
    ),
    sim_params={"n_citizens": 70, "n_timesteps": 25, "n_misinfo_agents": 10},
    recommended_strategies=[
        StrategyConfig(
            name="Transparent Authority",
            start_timestep=1,
            tone_multiplier=1.0,
            posts_per_timestep=3,
            multi_channel=True,
            tone_label="authoritative",
            color="#2196F3",
            description=(
                "Immediate transparent authoritative messaging. High frequency, "
                "multi-channel. Proactively addresses rumours with facts."
            ),
        ),
        StrategyConfig(
            name="Empathetic Trust Build",
            start_timestep=1,
            tone_multiplier=0.85,
            posts_per_timestep=2,
            multi_channel=True,
            tone_label="empathetic",
            color="#4CAF50",
            description=(
                "Prioritises emotional reassurance alongside factual guidance to "
                "counter fear-driven misinformation adoption."
            ),
        ),
        StrategyConfig(
            name="Delayed Official Response",
            start_timestep=8,
            tone_multiplier=0.75,
            posts_per_timestep=1,
            multi_channel=False,
            tone_label="neutral",
            color="#9E9E9E",
            description=(
                "Conservative response — government delays official communication "
                "during investigation. Demonstrates cost of silence."
            ),
        ),
    ],
)


# ── Registry ──────────────────────────────────────────────────────────────────

ALL_SCENARIOS: Dict[str, ScenarioCard] = {
    s.id: s
    for s in [
        SCENARIO_COVID_PANDEMIC,
        SCENARIO_DISEASE_OUTBREAK,
        SCENARIO_NATURAL_DISASTER,
        SCENARIO_BIOTERRORISM,
    ]
}


def get_scenario(scenario_id: str) -> ScenarioCard:
    """Return a ScenarioCard by its id slug.

    Raises KeyError if the id is not found.
    """
    if scenario_id not in ALL_SCENARIOS:
        raise KeyError(
            f"Unknown scenario: {scenario_id!r}. "
            f"Available: {list(ALL_SCENARIOS)}"
        )
    return ALL_SCENARIOS[scenario_id]


def list_scenarios() -> List[ScenarioCard]:
    """Return all available ScenarioCards."""
    return list(ALL_SCENARIOS.values())
