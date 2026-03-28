"""Tests for BeliefTracker and INTERVIEW action parsing."""
import pytest
from oasis_crisis_communi.belief_tracker import BeliefTracker, InterviewResponse


class TestInterviewResponseParsing:
    """Test that INTERVIEW responses are correctly parsed into belief scores."""

    def test_parse_positive_trust_response(self, belief_tracker):
        """Strongly positive responses should yield high belief scores."""
        response = (
            "I trust the official guidance completely. The government has been "
            "transparent and I believe their recommendations are accurate."
        )
        result = belief_tracker.parse_interview_response(
            agent_id=1, timestep=1, raw_response=response
        )
        assert isinstance(result, InterviewResponse)
        assert result.belief_score > 0.6, f"Expected >0.6, got {result.belief_score}"
        assert result.sentiment == "positive"

    def test_parse_negative_distrust_response(self, belief_tracker):
        """Distrust responses should yield low belief scores."""
        response = (
            "I distrust the official guidance. I don't believe what they're saying. "
            "The information seems misleading and I doubt its accuracy."
        )
        result = belief_tracker.parse_interview_response(
            agent_id=2, timestep=1, raw_response=response
        )
        assert result.belief_score < 0.5, f"Expected <0.5, got {result.belief_score}"

    def test_parse_neutral_uncertain_response(self, belief_tracker):
        """Uncertain responses should yield mid-range belief scores."""
        response = (
            "I'm not sure what to believe. The information is unclear and "
            "there are contradictory reports. I'm uncertain about the guidance."
        )
        result = belief_tracker.parse_interview_response(
            agent_id=3, timestep=1, raw_response=response
        )
        assert 0.3 <= result.belief_score <= 0.7, f"Expected 0.3-0.7, got {result.belief_score}"

    def test_belief_score_bounded_0_to_1(self, belief_tracker):
        """Belief scores must always be in [0, 1]."""
        test_responses = [
            "I trust trust trust trust believe agree fully and completely support",
            "I distrust distrust distrust doubt false misleading propaganda lie reject",
            "",
            "Maybe. Not sure. Could be.",
        ]
        for i, resp in enumerate(test_responses):
            result = belief_tracker.parse_interview_response(
                agent_id=i, timestep=1, raw_response=resp
            )
            assert 0.0 <= result.belief_score <= 1.0, (
                f"Score {result.belief_score} out of [0,1] for response: {resp[:50]}"
            )

    def test_parse_json_embedded_belief_score(self, belief_tracker):
        """If response contains JSON with belief_score, it should be used."""
        response = 'I have some views. {"belief_score": 0.82, "sentiment": "positive"}'
        result = belief_tracker.parse_interview_response(
            agent_id=5, timestep=2, raw_response=response
        )
        # Should blend JSON score with keyword analysis
        assert result.belief_score > 0.5, f"Expected >0.5 due to embedded 0.82, got {result.belief_score}"

    def test_returns_interview_response_dataclass(self, belief_tracker):
        """parse_interview_response returns an InterviewResponse with correct fields."""
        response = "I believe the guidance is correct."
        result = belief_tracker.parse_interview_response(
            agent_id=7, timestep=3, raw_response=response
        )
        assert result.agent_id == 7
        assert result.timestep == 3
        assert result.raw_response == response
        assert isinstance(result.keywords, list)

    def test_to_dict_structure(self, belief_tracker):
        """InterviewResponse.to_dict() should return structured dict."""
        response = "I trust the official guidance."
        result = belief_tracker.parse_interview_response(
            agent_id=1, timestep=1, raw_response=response
        )
        d = result.to_dict()
        required_keys = ["agent_id", "timestep", "question", "raw_response", "belief_score", "sentiment", "keywords"]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"


class TestBeliefTrackerAggregation:
    """Test BeliefTracker aggregation methods."""

    def test_population_alignment_at_timestep(self, belief_tracker):
        """get_population_alignment returns mean belief for a timestep."""
        responses = [
            ("I trust this", 0.8),  # positive
            ("I trust this too", 0.8),
            ("I'm uncertain", 0.5),
            ("I doubt this", 0.25),
        ]
        for i, (resp, _) in enumerate(responses):
            belief_tracker.parse_interview_response(
                agent_id=i, timestep=5, raw_response=resp
            )

        alignment = belief_tracker.get_population_alignment(5)
        assert 0.0 <= alignment <= 1.0, f"Alignment {alignment} out of [0,1]"
        assert alignment > 0.3, "Should be above 0.3 with mixed positive/neutral responses"

    def test_population_alignment_empty_timestep(self, belief_tracker):
        """Returns 0.0 for a timestep with no interviews."""
        alignment = belief_tracker.get_population_alignment(99)
        assert alignment == 0.0

    def test_agent_belief_trajectory_tracking(self, belief_tracker):
        """Agent belief trajectories are tracked across timesteps."""
        responses = [
            "I'm uncertain about the guidance.",
            "I'm starting to trust the official guidance more.",
            "I trust and believe the official guidance is accurate.",
        ]
        for t, resp in enumerate(responses, start=1):
            belief_tracker.parse_interview_response(
                agent_id=42, timestep=t, raw_response=resp
            )

        trajectory = belief_tracker.get_agent_belief_trajectory(42)
        assert len(trajectory) == 3
        assert all(0.0 <= s <= 1.0 for s in trajectory)

    def test_get_all_responses_returns_list_of_dicts(self, belief_tracker):
        """get_all_responses returns list of dicts."""
        for i in range(3):
            belief_tracker.parse_interview_response(
                agent_id=i, timestep=1, raw_response="I trust the guidance."
            )
        all_responses = belief_tracker.get_all_responses()
        assert isinstance(all_responses, list)
        assert len(all_responses) == 3
        assert all(isinstance(r, dict) for r in all_responses)

    def test_reset_clears_state(self, belief_tracker):
        """reset() clears all stored data."""
        belief_tracker.parse_interview_response(1, 1, "I trust this.")
        belief_tracker.reset()
        assert belief_tracker.get_all_responses() == []
        assert belief_tracker.get_population_alignment(1) == 0.0
