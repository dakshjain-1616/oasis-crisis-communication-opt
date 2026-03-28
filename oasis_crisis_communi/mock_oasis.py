"""
Mock OASIS implementation for testing and environments without camel-ai.

Provides the same interface as the CAMEL-AI OASIS social simulation framework,
allowing the optimizer to run without the full camel dependency.
"""

import random
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class ActionType(Enum):
    POST = "post"
    REPOST = "repost"
    INTERVIEW = "interview"
    FOLLOW = "follow"
    LIKE = "like"
    DO_NOTHING = "do_nothing"


@dataclass
class Message:
    """A social media message posted by an agent."""
    content: str
    author_id: int
    timestep: int
    credibility: float = 0.5
    is_misinformation: bool = False
    reposts: int = 0
    likes: int = 0


@dataclass
class InterviewResponse:
    """Structured response from an INTERVIEW action."""
    agent_id: int
    belief_score: float          # [0,1] where 1 = fully aligned with official guidance
    confidence: float            # [0,1] how confident the agent is
    reasoning: str               # Text explanation
    influenced_by: List[str] = field(default_factory=list)  # Sources that shaped belief

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "belief_score": round(self.belief_score, 4),
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "influenced_by": self.influenced_by,
        }


class SocialAgent:
    """
    Mock social agent representing a citizen in the simulation.

    Belief dynamics follow a bounded confidence model where agents
    update beliefs based on message credibility and social influence.
    """

    def __init__(
        self,
        agent_id: int,
        initial_belief: float = 0.5,
        susceptibility: float = 0.5,
        network_centrality: float = 0.5,
        agent_type: str = "citizen",
    ):
        self.agent_id = agent_id
        self.belief = initial_belief
        self.susceptibility = susceptibility  # How easily belief changes
        self.network_centrality = network_centrality
        self.agent_type = agent_type
        self.received_messages: List[Message] = []
        self.repost_history: List[Message] = []
        self._rng = random.Random(agent_id * 42)

    def receive_message(self, msg: Message) -> None:
        """Process an incoming message and update belief."""
        self.received_messages.append(msg)
        self._update_belief(msg)

    def _update_belief(self, msg: Message) -> None:
        """Update belief score based on message credibility."""
        if msg.is_misinformation:
            # Misinformation pulls belief toward 0
            delta = -msg.credibility * self.susceptibility * 0.15
        else:
            # Official content pulls belief toward 1
            delta = msg.credibility * self.susceptibility * 0.12

        # Add slight noise
        delta += self._rng.gauss(0, 0.01)

        # Bounded update
        self.belief = max(0.0, min(1.0, self.belief + delta))

    def will_repost(self, msg: Message) -> bool:
        """Decide whether to repost a message based on alignment with belief."""
        if msg.is_misinformation:
            # Low-belief agents more likely to repost misinformation
            repost_prob = (1.0 - self.belief) * 0.3 * self.network_centrality
        else:
            # High-belief agents more likely to repost official content
            repost_prob = self.belief * 0.25 * self.network_centrality
        return self._rng.random() < repost_prob

    def interview(self) -> InterviewResponse:
        """Conduct an INTERVIEW action — returns structured belief state."""
        confidence = abs(self.belief - 0.5) * 2  # Higher confidence at extremes

        if self.belief >= 0.7:
            reasoning = (
                f"Agent {self.agent_id} strongly trusts official guidance. "
                f"Received {len(self.received_messages)} messages."
            )
        elif self.belief >= 0.5:
            reasoning = (
                f"Agent {self.agent_id} moderately aligns with official guidance "
                f"after evaluating multiple sources."
            )
        elif self.belief >= 0.3:
            reasoning = (
                f"Agent {self.agent_id} is uncertain, influenced by some "
                f"contradictory information."
            )
        else:
            reasoning = (
                f"Agent {self.agent_id} doubts official guidance, heavily "
                f"influenced by misinformation."
            )

        sources = []
        if self.received_messages:
            official = sum(1 for m in self.received_messages if not m.is_misinformation)
            misinfo = len(self.received_messages) - official
            if official > 0:
                sources.append(f"official ({official} msgs)")
            if misinfo > 0:
                sources.append(f"misinformation ({misinfo} msgs)")

        return InterviewResponse(
            agent_id=self.agent_id,
            belief_score=round(self.belief, 4),
            confidence=round(confidence, 4),
            reasoning=reasoning,
            influenced_by=sources,
        )


class GovernmentAgent:
    """Special agent that posts official crisis communications."""

    def __init__(self, agent_id: int = 0, strategy=None):
        self.agent_id = agent_id
        self.strategy = strategy
        self.posted_messages: List[Message] = []

    def post(self, timestep: int, credibility: float = 0.7) -> List[Message]:
        """Generate official messages for this timestep."""
        if self.strategy is None:
            count = 1
            content_fn = lambda t: f"Official health guidance update [t={t}]."
        else:
            count = self.strategy.frequency
            content_fn = self.strategy.message_template
            credibility = self.strategy.credibility_score()

        messages = []
        for i in range(count):
            msg = Message(
                content=content_fn(timestep),
                author_id=self.agent_id,
                timestep=timestep,
                credibility=credibility,
                is_misinformation=False,
            )
            self.posted_messages.append(msg)
            messages.append(msg)
        return messages


class MisinformationAgent:
    """Agent that spreads contradictory / false information."""

    def __init__(self, agent_id: int, virality: float = 0.6):
        self.agent_id = agent_id
        self.virality = virality  # How persuasive/viral their content is
        self.posted_messages: List[Message] = []
        self._rng = random.Random(agent_id * 7)

    def post(self, timestep: int) -> List[Message]:
        """Generate misinformation messages."""
        templates = [
            f"BREAKING: Official advice is WRONG [t={timestep}] — don't trust them!",
            f"Experts LIED about this [t={timestep}]. The truth they're hiding...",
            f"Government cover-up exposed [t={timestep}]! Share before removed!",
            f"My doctor says the OPPOSITE of what officials claim [t={timestep}]",
            f"PROOF that official guidance is dangerous [t={timestep}] — stay safe!",
        ]
        content = self._rng.choice(templates)
        msg = Message(
            content=content,
            author_id=self.agent_id,
            timestep=timestep,
            credibility=self.virality,
            is_misinformation=True,
        )
        self.posted_messages.append(msg)
        return [msg]


class SocialNetwork:
    """
    Mock social network managing agent connections and message propagation.
    Uses a simple scale-free-like topology.
    """

    def __init__(self, n_citizens: int = 50, seed: int = 42):
        self._rng = random.Random(seed)
        self.n_citizens = n_citizens
        self._adjacency: Dict[int, List[int]] = {}
        self._build_network(n_citizens)

    def _build_network(self, n: int) -> None:
        """Build a preferential-attachment-style network."""
        # Each node connects to a few random others (simplified)
        for i in range(n):
            k = self._rng.randint(2, min(8, n - 1))
            neighbors = self._rng.sample([j for j in range(n) if j != i], k)
            self._adjacency[i] = neighbors

    def get_neighbors(self, agent_id: int) -> List[int]:
        return self._adjacency.get(agent_id, [])

    def centrality(self, agent_id: int) -> float:
        """Approximate centrality as normalized degree."""
        degree = len(self._adjacency.get(agent_id, []))
        return min(1.0, degree / 10.0)
