"""Tests for oracle evaluation module.

This module contains both unit tests and property-based tests for oracle evaluation.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from exp.oracle import (
    OracleConfig,
    OracleEvalResults,
    OracleResult,
)
from exp.run_oracle import compute_metrics, get_questions_for_eval

# =============================================================================
# Strategies for property-based testing
# =============================================================================

# Strategy for valid judge scores (1-5)
judge_scores = st.integers(min_value=1, max_value=5)

# Strategy for non-empty strings
non_empty_strings = st.text(min_size=1, max_size=200)

# Strategy for context strings
context_strings = st.text(min_size=1, max_size=500)

# Strategy for question strings
question_strings = st.text(min_size=1, max_size=200)

# Strategy for oracle response strings
response_strings = st.text(min_size=1, max_size=1000)

# Strategy for judge reasoning strings
reasoning_strings = st.text(min_size=1, max_size=500)

# Strategy for judge prompt strings (JSON format)
judge_prompt_strings = st.text(min_size=1, max_size=2000)

# Strategy for verbalizer prompt strings
verbalizer_prompt_strings = st.text(min_size=1, max_size=2000)


# =============================================================================
# Tests for OracleResult
# =============================================================================


class TestOracleResult:
    """Unit tests for OracleResult dataclass."""

    def test_creates_valid_result(self) -> None:
        """Test creating a valid OracleResult."""
        result = OracleResult(
            context="Test context",
            question="Test question?",
            oracle_response="Test response",
            judge_score=4,
            judge_reasoning="Good response",
            judge_prompt="Test user prompt",
            verbalizer_prompt="Test verbalizer prompt",
        )
        assert result.context == "Test context"
        assert result.question == "Test question?"
        assert result.oracle_response == "Test response"
        assert result.judge_score == 4
        assert result.judge_reasoning == "Good response"
        assert result.judge_prompt == "Test user prompt"
        assert result.verbalizer_prompt == "Test verbalizer prompt"

    def test_raises_for_invalid_score_too_low(self) -> None:
        """Test that OracleResult raises for score < 1."""
        with pytest.raises(AssertionError, match="Judge score must be between 1 and 5"):
            OracleResult(
                context="Test",
                question="Test?",
                oracle_response="Test",
                judge_score=0,
                judge_reasoning="Test",
                judge_prompt="Test user prompt",
                verbalizer_prompt="Test verbalizer prompt",
            )

    def test_raises_for_invalid_score_too_high(self) -> None:
        """Test that OracleResult raises for score > 5."""
        with pytest.raises(AssertionError, match="Judge score must be between 1 and 5"):
            OracleResult(
                context="Test",
                question="Test?",
                oracle_response="Test",
                judge_score=6,
                judge_reasoning="Test",
                judge_prompt="Test user prompt",
                verbalizer_prompt="Test verbalizer prompt",
            )

    def test_raises_for_empty_reasoning(self) -> None:
        """Test that OracleResult raises for empty reasoning."""
        with pytest.raises(AssertionError, match="Judge reasoning cannot be empty"):
            OracleResult(
                context="Test",
                question="Test?",
                oracle_response="Test",
                judge_score=3,
                judge_reasoning="",
                judge_prompt="Test user prompt",
                verbalizer_prompt="Test verbalizer prompt",
            )

    def test_raises_for_empty_response(self) -> None:
        """Test that OracleResult raises for empty response."""
        with pytest.raises(AssertionError, match="Oracle response cannot be empty"):
            OracleResult(
                context="Test",
                question="Test?",
                oracle_response="",
                judge_score=3,
                judge_reasoning="Test",
                judge_prompt="Test user prompt",
                verbalizer_prompt="Test verbalizer prompt",
            )

    def test_raises_for_empty_context(self) -> None:
        """Test that OracleResult raises for empty context."""
        with pytest.raises(AssertionError, match="Context cannot be empty"):
            OracleResult(
                context="",
                question="Test?",
                oracle_response="Test",
                judge_score=3,
                judge_reasoning="Test",
                judge_prompt="Test user prompt",
                verbalizer_prompt="Test verbalizer prompt",
            )

    def test_raises_for_empty_question(self) -> None:
        """Test that OracleResult raises for empty question."""
        with pytest.raises(AssertionError, match="Question cannot be empty"):
            OracleResult(
                context="Test",
                question="",
                oracle_response="Test",
                judge_score=3,
                judge_reasoning="Test",
                judge_prompt="Test user prompt",
                verbalizer_prompt="Test verbalizer prompt",
            )

    def test_raises_for_empty_judge_prompt(self) -> None:
        """Test that OracleResult raises for empty judge prompt."""
        with pytest.raises(AssertionError, match="Judge prompt cannot be empty"):
            OracleResult(
                context="Test",
                question="Test?",
                oracle_response="Test",
                judge_score=3,
                judge_reasoning="Test",
                judge_prompt="",
                verbalizer_prompt="Test verbalizer prompt",
            )

    def test_raises_for_empty_verbalizer_prompt(self) -> None:
        """Test that OracleResult raises for empty verbalizer prompt."""
        with pytest.raises(AssertionError, match="Verbalizer prompt cannot be empty"):
            OracleResult(
                context="Test",
                question="Test?",
                oracle_response="Test",
                judge_score=3,
                judge_reasoning="Test",
                judge_prompt="Test user prompt",
                verbalizer_prompt="",
            )

    @given(
        context=context_strings,
        question=question_strings,
        response=response_strings,
        score=judge_scores,
        reasoning=reasoning_strings,
        prompt=judge_prompt_strings,
        verbalizer_prompt=verbalizer_prompt_strings,
    )
    @settings(max_examples=50)
    def test_property_valid_result_creation(
        self,
        context: str,
        question: str,
        response: str,
        score: int,
        reasoning: str,
        prompt: str,
        verbalizer_prompt: str,
    ) -> None:
        """Property: Can create OracleResult with any valid inputs."""
        result = OracleResult(
            context=context,
            question=question,
            oracle_response=response,
            judge_score=score,
            judge_reasoning=reasoning,
            judge_prompt=prompt,
            verbalizer_prompt=verbalizer_prompt,
        )
        assert result.context == context
        assert result.question == question
        assert result.oracle_response == response
        assert result.judge_score == score
        assert result.judge_reasoning == reasoning
        assert result.judge_prompt == prompt
        assert result.verbalizer_prompt == verbalizer_prompt
        assert 1 <= result.judge_score <= 5


# =============================================================================
# Tests for OracleEvalResults
# =============================================================================


class TestOracleEvalResults:
    """Unit tests for OracleEvalResults dataclass."""

    def test_creates_valid_results(self) -> None:
        """Test creating OracleEvalResults with valid data."""
        config = OracleConfig()
        results = [
            OracleResult(
                context="Context 1",
                question="Question 1?",
                oracle_response="Response 1",
                judge_score=4,
                judge_reasoning="Good",
                judge_prompt="Test user prompt",
                verbalizer_prompt="Test verbalizer prompt",
            ),
            OracleResult(
                context="Context 2",
                question="Question 2?",
                oracle_response="Response 2",
                judge_score=5,
                judge_reasoning="Excellent",
                judge_prompt="Test user prompt",
                verbalizer_prompt="Test verbalizer prompt",
            ),
        ]
        eval_results = OracleEvalResults(config=config, results=results)
        assert len(eval_results.results) == 2
        assert eval_results.config == config

    def test_mean_score_calculation(self) -> None:
        """Test mean_score calculation."""
        config = OracleConfig()
        results = [
            OracleResult(
                context="C1",
                question="Q1?",
                oracle_response="R1",
                judge_score=3,
                judge_reasoning="OK",
                judge_prompt="Test user prompt",
                verbalizer_prompt="Test verbalizer prompt",
            ),
            OracleResult(
                context="C2",
                question="Q2?",
                oracle_response="R2",
                judge_score=5,
                judge_reasoning="Great",
                judge_prompt="Test user prompt",
                verbalizer_prompt="Test verbalizer prompt",
            ),
            OracleResult(
                context="C3",
                question="Q3?",
                oracle_response="R3",
                judge_score=4,
                judge_reasoning="Good",
                judge_prompt="Test user prompt",
                verbalizer_prompt="Test verbalizer prompt",
            ),
        ]
        eval_results = OracleEvalResults(config=config, results=results)
        mean = eval_results.mean_score()
        assert mean == pytest.approx(4.0, abs=0.01)

    def test_raises_for_empty_results(self) -> None:
        """Test that mean_score raises for empty results."""
        config = OracleConfig()
        eval_results = OracleEvalResults(config=config, results=[])
        with pytest.raises(ValueError, match="No scores to compute mean score"):
            eval_results.mean_score()

    @given(
        scores=st.lists(judge_scores, min_size=1, max_size=20),
    )
    @settings(max_examples=50)
    def test_property_mean_score_in_range(
        self,
        scores: list[int],
    ) -> None:
        """Property: Mean score is always between 1 and 5."""
        config = OracleConfig()
        results = [
            OracleResult(
                context=f"Context {i}",
                question=f"Question {i}?",
                oracle_response=f"Response {i}",
                judge_score=score,
                judge_reasoning=f"Reasoning {i}",
                judge_prompt="Test user prompt",
                verbalizer_prompt="Test verbalizer prompt",
            )
            for i, score in enumerate(scores)
        ]
        eval_results = OracleEvalResults(config=config, results=results)
        mean = eval_results.mean_score()
        assert 1.0 <= mean <= 5.0

    @given(
        scores=st.lists(judge_scores, min_size=1, max_size=10),
    )
    @settings(max_examples=50)
    def test_property_mean_score_equals_sum_over_count(
        self,
        scores: list[int],
    ) -> None:
        """Property: Mean score equals sum of scores divided by count."""
        config = OracleConfig()
        results = [
            OracleResult(
                context=f"Context {i}",
                question=f"Question {i}?",
                oracle_response=f"Response {i}",
                judge_score=score,
                judge_reasoning=f"Reasoning {i}",
                judge_prompt="Test user prompt",
                verbalizer_prompt="Test verbalizer prompt",
            )
            for i, score in enumerate(scores)
        ]
        eval_results = OracleEvalResults(config=config, results=results)
        mean = eval_results.mean_score()
        expected_mean = sum(scores) / len(scores)
        assert mean == pytest.approx(expected_mean, abs=0.01)


# =============================================================================
# Tests for compute_metrics
# =============================================================================


class TestComputeMetrics:
    """Unit tests for compute_metrics function."""

    def test_computes_basic_metrics(self) -> None:
        """Test basic metric computation."""
        config = OracleConfig()
        results = [
            OracleResult(
                context="Context 1",
                question="Question 1?",
                oracle_response="Response 1",
                judge_score=4,
                judge_reasoning="Good",
                judge_prompt="Test user prompt",
                verbalizer_prompt="Test verbalizer prompt",
            ),
            OracleResult(
                context="Context 2",
                question="Question 2?",
                oracle_response="Response 2",
                judge_score=5,
                judge_reasoning="Excellent",
                judge_prompt="Test user prompt",
                verbalizer_prompt="Test verbalizer prompt",
            ),
        ]
        eval_results = OracleEvalResults(config=config, results=results)
        metrics = compute_metrics(eval_results)

        assert metrics["total_queries"] == 2
        assert metrics["unique_contexts"] == 2
        assert metrics["unique_questions"] == 2
        assert metrics["mean_score"] == pytest.approx(4.5, abs=0.01)
        assert metrics["min_score"] == 4
        assert metrics["max_score"] == 5
        assert metrics["score_distribution"]["4"] == 1
        assert metrics["score_distribution"]["5"] == 1

    def test_raises_for_empty_results(self) -> None:
        """Test that compute_metrics raises for empty results."""
        config = OracleConfig()
        eval_results = OracleEvalResults(config=config, results=[])
        with pytest.raises(AssertionError, match="Results cannot be empty"):
            compute_metrics(eval_results)

    @given(
        num_results=st.integers(min_value=1, max_value=10),
        scores=st.lists(judge_scores, min_size=1, max_size=10),
    )
    @settings(max_examples=50)
    def test_property_metrics_consistent(
        self,
        num_results: int,
        scores: list[int],
    ) -> None:
        """Property: Metrics are consistent with input data."""
        # Use provided scores or generate more if needed
        if len(scores) < num_results:
            scores = scores * ((num_results // len(scores)) + 1)
        scores = scores[:num_results]

        config = OracleConfig()
        results = [
            OracleResult(
                context=f"Context {i}",
                question=f"Question {i}?",
                oracle_response=f"Response {i}",
                judge_score=score,
                judge_reasoning=f"Reasoning {i}",
                judge_prompt="Test user prompt",
                verbalizer_prompt="Test verbalizer prompt",
            )
            for i, score in enumerate(scores)
        ]
        eval_results = OracleEvalResults(config=config, results=results)
        metrics = compute_metrics(eval_results)

        # Check consistency
        assert metrics["total_queries"] == num_results
        assert metrics["mean_score"] == pytest.approx(
            sum(scores) / len(scores), abs=0.01
        )
        assert metrics["min_score"] == min(scores)
        assert metrics["max_score"] == max(scores)
        assert sum(metrics["score_distribution"].values()) == num_results

    @given(
        scores=st.lists(judge_scores, min_size=1, max_size=20),
    )
    @settings(max_examples=50)
    def test_property_score_distribution_complete(
        self,
        scores: list[int],
    ) -> None:
        """Property: Score distribution covers all scores 1-5."""
        config = OracleConfig()
        results = [
            OracleResult(
                context=f"Context {i}",
                question=f"Question {i}?",
                oracle_response=f"Response {i}",
                judge_score=score,
                judge_reasoning=f"Reasoning {i}",
                judge_prompt="Test user prompt",
                verbalizer_prompt="Test verbalizer prompt",
            )
            for i, score in enumerate(scores)
        ]
        eval_results = OracleEvalResults(config=config, results=results)
        metrics = compute_metrics(eval_results)

        # All scores 1-5 should be in distribution (even if count is 0)
        dist = metrics["score_distribution"]
        assert all(str(i) in dist for i in range(1, 6))
        assert sum(dist.values()) == len(scores)


# =============================================================================
# Tests for get_questions_for_eval
# =============================================================================


class TestGetQuestionsForEval:
    """Unit tests for get_questions_for_eval function."""

    def test_returns_train_questions(self) -> None:
        """Test getting train questions."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "questions": {
                    "split": "train",
                    "n_questions": None,
                }
            }
        )
        cfg = OmegaConf.structured(cfg)
        questions = get_questions_for_eval(cfg)
        assert len(questions) > 0
        assert all(isinstance(q, str) for q in questions)

    def test_returns_val_questions(self) -> None:
        """Test getting val questions."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "questions": {
                    "split": "val",
                    "n_questions": None,
                }
            }
        )
        cfg = OmegaConf.structured(cfg)
        questions = get_questions_for_eval(cfg)
        assert len(questions) > 0
        assert all(isinstance(q, str) for q in questions)

    def test_limits_questions(self) -> None:
        """Test limiting number of questions."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "questions": {
                    "split": "train",
                    "n_questions": 3,
                }
            }
        )
        cfg = OmegaConf.structured(cfg)
        questions = get_questions_for_eval(cfg)
        assert len(questions) == 3

    def test_raises_for_invalid_split(self) -> None:
        """Test that invalid split raises error."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "questions": {
                    "split": "invalid",
                    "n_questions": None,
                }
            }
        )
        cfg = OmegaConf.structured(cfg)
        with pytest.raises(ValueError, match="Unknown question split"):
            get_questions_for_eval(cfg)

    @given(
        n_questions=st.integers(min_value=1, max_value=20) | st.none(),
    )
    @settings(max_examples=30)
    def test_property_returns_non_empty_list(
        self,
        n_questions: int | None,
    ) -> None:
        """Property: Always returns non-empty list of strings."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "questions": {
                    "split": "train",
                    "n_questions": n_questions,
                }
            }
        )
        cfg = OmegaConf.structured(cfg)
        questions = get_questions_for_eval(cfg)
        assert len(questions) > 0
        assert all(isinstance(q, str) for q in questions)
        assert all(len(q) > 0 for q in questions)

    @given(
        n_questions=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=30)
    def test_property_respects_n_questions_limit(
        self,
        n_questions: int,
    ) -> None:
        """Property: Returns at most n_questions when specified."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "questions": {
                    "split": "train",
                    "n_questions": n_questions,
                }
            }
        )
        cfg = OmegaConf.structured(cfg)
        questions = get_questions_for_eval(cfg)
        assert len(questions) <= n_questions
