"""Belief tracking via simulated INTERVIEW actions."""
from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class InterviewResponse:
    """Structured response from an INTERVIEW action."""
    agent_id: int
    timestep: int
    question: str
    raw_response: str
    belief_score: float  # 0.0 = disbelief, 1.0 = full belief
    sentiment: str  # "positive", "neutral", "negative"
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "timestep": self.timestep,
            "question": self.question,
            "raw_response": self.raw_response,
            "belief_score": round(self.belief_score, 4),
            "sentiment": self.sentiment,
            "keywords": self.keywords,
        }


class BeliefTracker:
    """
    Tracks agent beliefs using INTERVIEW-style queries.

    In real OASIS mode, this sends ActionType.INTERVIEW messages via the Channel.
    In mock mode, it uses the simulation's internal belief state.
    """

    INTERVIEW_QUESTION = (
        "Regarding the ongoing public health crisis, do you believe the "
        "official government guidance is accurate and should be followed? "
        "Please respond with: your overall opinion (trust/uncertain/distrust), "
        "and a brief explanation of why."
    )

    POSITIVE_KEYWORDS = [
        "trust", "believe", "accurate", "follow", "official", "reliable",
        "credible", "confident", "agree", "correct", "helpful", "important",
        "safe", "effective", "support", "endorse", "valid",
    ]

    NEGATIVE_KEYWORDS = [
        "distrust", "doubt", "false", "misleading", "propaganda", "lie",
        "conspiracy", "uncertain", "disagree", "wrong", "harmful", "fake",
        "manipulate", "ignore", "reject", "suspicious", "corrupt",
    ]

    def __init__(self):
        self._history: List[InterviewResponse] = []
        self._agent_beliefs: Dict[int, List[float]] = {}

    def parse_interview_response(
        self,
        agent_id: int,
        timestep: int,
        raw_response: str,
        question: Optional[str] = None,
    ) -> InterviewResponse:
        """
        Parse an INTERVIEW action response into a structured belief score.

        Looks for explicit trust indicators first, then falls back to
        keyword-weighted scoring.
        """
        question = question or self.INTERVIEW_QUESTION
        lower = raw_response.lower()

        # Check for explicit trust keywords
        sentiment = "neutral"
        base_score = 0.5

        if any(kw in lower for kw in ["i trust", "i believe", "i agree", "yes, i"]):
            sentiment = "positive"
            base_score = 0.75
        elif any(kw in lower for kw in ["i distrust", "i don't believe", "i disagree", "no, i"]):
            sentiment = "negative"
            base_score = 0.25
        elif "uncertain" in lower or "not sure" in lower or "unclear" in lower:
            sentiment = "neutral"
            base_score = 0.5

        # Extract JSON if present
        json_match = re.search(r'\{[^}]+\}', raw_response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if "belief_score" in data:
                    base_score = float(data["belief_score"])
                if "sentiment" in data:
                    sentiment = data["sentiment"]
            except (json.JSONDecodeError, ValueError):
                pass

        # Keyword weighting (word-boundary match to avoid substrings like "lie" in "belief")
        pos_hits = sum(1 for kw in self.POSITIVE_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', lower))
        neg_hits = sum(1 for kw in self.NEGATIVE_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', lower))

        total_hits = pos_hits + neg_hits
        if total_hits > 0:
            keyword_score = pos_hits / total_hits
            # Blend base score with keyword score
            base_score = 0.6 * base_score + 0.4 * keyword_score

        # Clamp to [0, 1]
        belief_score = max(0.0, min(1.0, base_score))

        # Extract found keywords
        found_keywords = [kw for kw in self.POSITIVE_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', lower)]
        found_keywords += [kw for kw in self.NEGATIVE_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', lower)]

        response = InterviewResponse(
            agent_id=agent_id,
            timestep=timestep,
            question=question,
            raw_response=raw_response,
            belief_score=belief_score,
            sentiment=sentiment,
            keywords=found_keywords[:10],
        )

        self._history.append(response)
        if agent_id not in self._agent_beliefs:
            self._agent_beliefs[agent_id] = []
        self._agent_beliefs[agent_id].append(belief_score)

        return response

    def get_population_alignment(self, timestep: int) -> float:
        """
        Get the average belief alignment score for a specific timestep.
        Returns value 0-1.
        """
        timestep_responses = [r for r in self._history if r.timestep == timestep]
        if not timestep_responses:
            return 0.0
        return sum(r.belief_score for r in timestep_responses) / len(timestep_responses)

    def get_agent_belief_trajectory(self, agent_id: int) -> List[float]:
        """Get the belief trajectory for a specific agent."""
        return self._agent_beliefs.get(agent_id, [])

    def get_all_responses(self) -> List[dict]:
        """Return all interview responses as list of dicts."""
        return [r.to_dict() for r in self._history]

    def reset(self):
        """Reset tracker state."""
        self._history.clear()
        self._agent_beliefs.clear()
