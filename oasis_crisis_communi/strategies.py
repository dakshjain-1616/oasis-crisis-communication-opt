"""Communication strategy definitions for crisis simulation."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StrategyConfig:
    """Configuration for a government communication strategy."""
    name: str
    # Timestep at which government begins posting (1-indexed)
    start_timestep: int = 1
    # Tone multiplier: 1.0=authoritative, 0.85=empathetic, 0.6=neutral
    tone_multiplier: float = 1.0
    # Number of posts government makes per timestep
    posts_per_timestep: int = 2
    # Whether to use multi-channel approach (amplifies reach)
    multi_channel: bool = False
    # Message template tokens for this strategy
    tone_label: str = "authoritative"
    # Color for visualization
    color: str = "#1f77b4"
    # Description for reports
    description: str = ""

    @property
    def frequency(self) -> int:
        """Alias for posts_per_timestep (used by GovernmentAgent)."""
        return self.posts_per_timestep

    def credibility_score(self) -> float:
        """Return a credibility multiplier in [0, 1] based on tone and timing."""
        base = {
            "authoritative": 0.80,
            "empathetic": 0.70,
            "neutral": 0.60,
            "reactive": 0.65,
        }.get(self.tone_label, 0.70)
        # Penalize strategies that start late
        timing_penalty = max(0.0, (self.start_timestep - 1) * 0.02)
        return max(0.0, min(1.0, base - timing_penalty))

    def message_template(self, timestep: int) -> str:
        """Generate a message string for this timestep."""
        templates = {
            "authoritative": (
                f"[T{timestep}] OFFICIAL: Follow the latest evidence-based guidance. "
                f"Timestep {timestep} update from health authorities."
            ),
            "empathetic": (
                f"[T{timestep}] We care about your wellbeing. "
                f"Here is the guidance for timestep {timestep}: stay safe and connected."
            ),
            "neutral": (
                f"[T{timestep}] Information update for timestep {timestep}: "
                f"see official channels for current recommendations."
            ),
            "reactive": (
                f"[T{timestep}] Correcting misinformation at timestep {timestep}: "
                f"official facts and clarifications follow."
            ),
        }
        return templates.get(
            self.tone_label,
            f"Official update at timestep {timestep} — follow guidance.",
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start_timestep": self.start_timestep,
            "tone_multiplier": self.tone_multiplier,
            "posts_per_timestep": self.posts_per_timestep,
            "multi_channel": self.multi_channel,
            "tone_label": self.tone_label,
            "color": self.color,
            "description": self.description,
        }


class CommunicationStrategy:
    """Factory for predefined communication strategies."""

    @staticmethod
    def early_authoritative() -> StrategyConfig:
        return StrategyConfig(
            name="Early & Authoritative",
            start_timestep=1,
            tone_multiplier=1.0,
            posts_per_timestep=2,
            multi_channel=False,
            tone_label="authoritative",
            color="#2196F3",
            description=(
                "Government posts from timestep 1 using authoritative, "
                "fact-based messaging. Medium posting frequency."
            ),
        )

    @staticmethod
    def empathetic_frequent() -> StrategyConfig:
        return StrategyConfig(
            name="Empathetic & Frequent",
            start_timestep=1,
            tone_multiplier=0.85,
            posts_per_timestep=3,
            multi_channel=True,
            tone_label="empathetic",
            color="#4CAF50",
            description=(
                "Government posts from timestep 1 with empathetic, "
                "community-focused messaging. High frequency, multi-channel."
            ),
        )

    @staticmethod
    def late_reactive() -> StrategyConfig:
        return StrategyConfig(
            name="Late & Reactive",
            start_timestep=8,
            tone_multiplier=1.0,
            posts_per_timestep=1,
            multi_channel=False,
            tone_label="reactive",
            color="#F44336",
            description=(
                "Government waits until misinformation is established "
                "before responding. Single-channel, low frequency."
            ),
        )

    @staticmethod
    def all_defaults() -> list:
        """Return the three default comparison strategies."""
        return [
            CommunicationStrategy.early_authoritative(),
            CommunicationStrategy.empathetic_frequent(),
            CommunicationStrategy.late_reactive(),
        ]


# ── Named strategy constants used by tests ────────────────────────────────────

STRATEGY_EARLY_AUTHORITATIVE = StrategyConfig(
    name="Early Authoritative",
    start_timestep=1,
    tone_multiplier=1.0,
    posts_per_timestep=2,
    multi_channel=False,
    tone_label="authoritative",
    color="#2196F3",
    description=(
        "Government begins posting from timestep 1 using authoritative, "
        "evidence-based messaging. Standard frequency."
    ),
)

STRATEGY_EMPATHETIC_COMMUNITY = StrategyConfig(
    name="Empathetic Community",
    start_timestep=1,
    tone_multiplier=0.85,
    posts_per_timestep=3,
    multi_channel=True,
    tone_label="empathetic",
    color="#4CAF50",
    description=(
        "Early, high-frequency empathetic messaging across multiple channels, "
        "prioritising community trust-building."
    ),
)

STRATEGY_DELAYED_NEUTRAL = StrategyConfig(
    name="Delayed Neutral",
    start_timestep=8,
    tone_multiplier=0.6,
    posts_per_timestep=1,
    multi_channel=False,
    tone_label="neutral",
    color="#F44336",
    description=(
        "Government waits until misinformation is established before responding "
        "with low-frequency, neutral-tone single-channel posts."
    ),
)

DEFAULT_STRATEGIES = [
    STRATEGY_EARLY_AUTHORITATIVE,
    STRATEGY_EMPATHETIC_COMMUNITY,
    STRATEGY_DELAYED_NEUTRAL,
]
