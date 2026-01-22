"""Tests for core/data.py - dataset loading and message processing.

This module contains property-based tests for the message cleaning logic
used in conversation dataset preparation.
"""

from __future__ import annotations

from typing import Any

from hypothesis import given
from hypothesis import strategies as st

# =============================================================================
# Strategies for generating test data
# =============================================================================

# Strategy for generating chat message roles
valid_roles = st.sampled_from(["user", "assistant", "system"])
mapped_roles = st.sampled_from(["environment"])  # Roles that get remapped
all_roles = st.one_of(valid_roles, mapped_roles)

# Strategy for message content
message_content = st.text(min_size=0, max_size=100)


@st.composite
def chat_message(draw: st.DrawFn) -> dict[str, str]:
    """Generate a chat message with role and content."""
    role = draw(all_roles)
    content = draw(message_content)
    return {"role": role, "content": content}


# Strategy for a list of chat messages (a conversation)
chat_messages = st.lists(chat_message(), min_size=0, max_size=10)


# =============================================================================
# Message cleaning logic (extracted from core/data.py for testing)
# =============================================================================


def clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Replicate the message cleaning logic from core/data.py.

    This is extracted here to test the cleaning invariants independently
    of tokenization. The actual implementation lives in the tokenize_conversation
    function in core/data.py.
    """
    role_map = {"environment": "user"}
    cleaned_messages: list[dict[str, str]] = []

    for msg in messages:
        role = msg.get("role", "user")
        role = role_map.get(role, role)  # Map unsupported roles
        content = msg.get("content", "")
        if not content:  # Skip empty messages
            continue
        # Merge consecutive messages with the same role
        if cleaned_messages and cleaned_messages[-1]["role"] == role:
            cleaned_messages[-1]["content"] += "\n\n" + content
        else:
            cleaned_messages.append({"role": role, "content": content})

    # Remove leading assistant messages
    while cleaned_messages and cleaned_messages[0]["role"] == "assistant":
        cleaned_messages.pop(0)

    return cleaned_messages


# =============================================================================
# Unit tests for message cleaning
# =============================================================================


class TestMessageCleaning:
    """Unit tests for message cleaning logic."""

    def test_empty_messages_list(self) -> None:
        """Empty input should produce empty output."""
        result = clean_messages([])
        assert result == []

    def test_single_user_message(self) -> None:
        """Single user message should pass through unchanged."""
        messages = [{"role": "user", "content": "Hello"}]
        result = clean_messages(messages)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_removes_empty_content(self) -> None:
        """Messages with empty content should be removed."""
        messages = [
            {"role": "user", "content": ""},
            {"role": "user", "content": "Hello"},
        ]
        result = clean_messages(messages)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_merges_consecutive_same_role(self) -> None:
        """Consecutive messages with same role should be merged."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
        ]
        result = clean_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Hello" in result[0]["content"]
        assert "World" in result[0]["content"]

    def test_maps_environment_to_user(self) -> None:
        """Environment role should be mapped to user."""
        messages = [{"role": "environment", "content": "System info"}]
        result = clean_messages(messages)
        assert result == [{"role": "user", "content": "System info"}]

    def test_removes_leading_assistant(self) -> None:
        """Leading assistant messages should be removed."""
        messages = [
            {"role": "assistant", "content": "I am an assistant"},
            {"role": "user", "content": "Hello"},
        ]
        result = clean_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_preserves_system_messages(self) -> None:
        """System messages should be preserved."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = clean_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_typical_conversation(self) -> None:
        """Test a typical multi-turn conversation."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "Thanks!"},
        ]
        result = clean_messages(messages)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"


# =============================================================================
# Property-based tests for message cleaning
# =============================================================================


class TestMessageCleaningProperties:
    """Property-based tests for message cleaning invariants."""

    @given(chat_messages)
    def test_no_consecutive_same_role(self, messages: list[dict[str, str]]) -> None:
        """After cleaning, no two consecutive messages should have the same role."""
        cleaned = clean_messages(messages)

        for i in range(len(cleaned) - 1):
            assert cleaned[i]["role"] != cleaned[i + 1]["role"], (
                f"Consecutive messages at {i} and {i + 1} have same role: "
                f"{cleaned[i]['role']}"
            )

    @given(chat_messages)
    def test_no_empty_content(self, messages: list[dict[str, str]]) -> None:
        """After cleaning, no message should have empty content."""
        cleaned = clean_messages(messages)

        for i, msg in enumerate(cleaned):
            assert msg["content"], f"Message at index {i} has empty content"

    @given(chat_messages)
    def test_never_starts_with_assistant(self, messages: list[dict[str, str]]) -> None:
        """After cleaning, conversation should never start with assistant role."""
        cleaned = clean_messages(messages)

        if cleaned:
            assert (
                cleaned[0]["role"] != "assistant"
            ), "Conversation starts with assistant role"

    @given(chat_messages)
    def test_environment_role_mapped(self, messages: list[dict[str, str]]) -> None:
        """Environment role should be mapped to user in cleaned messages."""
        cleaned = clean_messages(messages)

        for msg in cleaned:
            assert (
                msg["role"] != "environment"
            ), "Environment role was not mapped to user"

    @given(chat_messages)
    def test_only_valid_roles(self, messages: list[dict[str, str]]) -> None:
        """Cleaned messages should only contain user, assistant, or system roles."""
        cleaned = clean_messages(messages)
        valid_roles_set = {"user", "assistant", "system"}

        for msg in cleaned:
            assert (
                msg["role"] in valid_roles_set
            ), f"Invalid role '{msg['role']}' in cleaned messages"

    @given(
        st.lists(
            st.fixed_dictionaries(
                {
                    "role": st.just("user"),
                    "content": st.text(min_size=1, max_size=50),
                }
            ),
            min_size=2,
            max_size=5,
        )
    )
    def test_consecutive_same_role_merged(self, messages: list[dict[str, str]]) -> None:
        """Multiple consecutive user messages should be merged into one."""
        cleaned = clean_messages(messages)

        # All user messages with non-empty content should merge into one
        assert len(cleaned) == 1
        assert cleaned[0]["role"] == "user"

        # The merged content should contain all original content
        for msg in messages:
            assert msg["content"] in cleaned[0]["content"]

    @given(
        st.lists(
            st.fixed_dictionaries(
                {
                    "role": st.just("assistant"),
                    "content": st.text(min_size=1, max_size=50),
                }
            ),
            min_size=1,
            max_size=5,
        )
    )
    def test_all_assistant_messages_removed(
        self, messages: list[dict[str, str]]
    ) -> None:
        """If all messages are assistant messages, result should be empty."""
        cleaned = clean_messages(messages)
        assert len(cleaned) == 0

    @given(chat_messages)
    def test_content_preserved(self, messages: list[dict[str, str]]) -> None:
        """Non-empty content from non-leading-assistant messages should be preserved."""
        cleaned = clean_messages(messages)

        # Collect all non-empty content that should be preserved
        # (excluding leading assistant messages)
        role_map = {"environment": "user"}
        expected_contents: list[str] = []
        found_non_assistant = False

        for msg in messages:
            role = msg.get("role", "user")
            role = role_map.get(role, role)
            content = msg.get("content", "")

            if not content:
                continue

            if role != "assistant":
                found_non_assistant = True

            # Only include content after we've found a non-assistant message
            if found_non_assistant:
                expected_contents.append(content)

        # Check all expected content is in the cleaned messages
        all_cleaned_content = " ".join(m["content"] for m in cleaned)
        for content in expected_contents:
            assert content in all_cleaned_content
