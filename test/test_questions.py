"""Tests for core/questions.py - activation oracle question sets."""

from __future__ import annotations

from core.questions import TRAIN_QUESTIONS, VAL_QUESTIONS, to_chat_message


class TestQuestionLists:
    """Tests for question list constants."""

    def test_train_questions_not_empty(self) -> None:
        """Train questions should have content."""
        assert len(TRAIN_QUESTIONS) > 0

    def test_val_questions_not_empty(self) -> None:
        """Val questions should have content."""
        assert len(VAL_QUESTIONS) > 0

    def test_all_train_questions_are_strings(self) -> None:
        """All train questions should be strings."""
        for q in TRAIN_QUESTIONS:
            assert isinstance(q, str)

    def test_all_val_questions_are_strings(self) -> None:
        """All val questions should be strings."""
        for q in VAL_QUESTIONS:
            assert isinstance(q, str)

    def test_no_overlap(self) -> None:
        """Train and val questions should not overlap."""
        train_set = set(TRAIN_QUESTIONS)
        val_set = set(VAL_QUESTIONS)
        assert train_set.isdisjoint(val_set)

    def test_no_empty_questions(self) -> None:
        """No questions should be empty strings."""
        for q in TRAIN_QUESTIONS + VAL_QUESTIONS:
            assert q.strip() != ""


class TestToChatMessage:
    """Tests for to_chat_message helper."""

    def test_returns_list(self) -> None:
        """Should return a list."""
        result = to_chat_message("What is the topic?")
        assert isinstance(result, list)

    def test_single_message(self) -> None:
        """Should return exactly one message."""
        result = to_chat_message("Test question")
        assert len(result) == 1

    def test_user_role(self) -> None:
        """Message should have user role."""
        result = to_chat_message("Test")
        assert result[0]["role"] == "user"

    def test_content_matches(self) -> None:
        """Content should match the input question."""
        question = "What is the model thinking?"
        result = to_chat_message(question)
        assert result[0]["content"] == question

    def test_empty_string(self) -> None:
        """Should handle empty string."""
        result = to_chat_message("")
        assert result == [{"role": "user", "content": ""}]
