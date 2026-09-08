"""Prompt-construction tests.

Citation numbering is the fragile part: the `[n]` markers the model is told to
use must line up exactly with the citations shown in the UI. If context is
truncated but citations are not, the answer cites sources the model never saw.
"""

from __future__ import annotations

from uuid import uuid4

from app.db.models import MessageRecord, SearchHit
from app.rag.prompt import (
    NO_CONTEXT_ANSWER,
    build_answer_prompt,
    build_condense_prompt,
    clean_condensed,
    format_context,
    needs_condensing,
)


def _hit(
    content: str, *, filename: str = "guide.pdf", page: int | None = 3, score: float = 0.9
) -> SearchHit:
    return SearchHit(
        chunk_id=uuid4(),
        document_id=uuid4(),
        filename=filename,
        page=page,
        content=content,
        score=score,
    )


def test_context_is_numbered_from_one() -> None:
    context, used = format_context(
        [_hit("First passage."), _hit("Second passage."), _hit("Third passage.")],
        max_chars=10_000,
    )
    assert "[1] Source: guide.pdf, page 3" in context
    assert "[2] Source: guide.pdf, page 3" in context
    assert "[3] Source: guide.pdf, page 3" in context
    assert len(used) == 3


def test_context_omits_the_page_when_there_is_none() -> None:
    context, _ = format_context([_hit("Body.", page=None)], max_chars=1000)
    assert "Source: guide.pdf\n" in context
    assert "page" not in context


def test_truncation_keeps_context_and_citations_in_step() -> None:
    hits = [_hit("x" * 400, filename=f"doc{i}.pdf") for i in range(5)]
    context, used = format_context(hits, max_chars=900)

    assert len(used) < len(hits), "expected truncation"
    # Every returned citation must have a matching marker, and no marker may
    # refer to a passage that was dropped.
    for i in range(1, len(used) + 1):
        assert f"[{i}] Source:" in context
    assert f"[{len(used) + 1}] Source:" not in context


def test_the_first_passage_is_always_included() -> None:
    """A single oversized hit must not yield an empty context and a false refusal."""
    context, used = format_context([_hit("y" * 5000)], max_chars=100)
    assert len(used) == 1
    assert context


def test_no_hits_gives_empty_context() -> None:
    context, used = format_context([], max_chars=1000)
    assert context == ""
    assert used == []


def test_answer_prompt_carries_question_and_context() -> None:
    prompt = build_answer_prompt("  What is the refund window?  ", "[1] ...")
    assert "What is the refund window?" in prompt
    assert "[1] ..." in prompt
    assert "  What is" not in prompt  # question is stripped


def test_condense_prompt_includes_the_transcript() -> None:
    history = [
        MessageRecord(role="user", content="Tell me about the warranty."),
        MessageRecord(role="assistant", content="It lasts 24 months."),
    ]
    prompt = build_condense_prompt(history, "And the second one?")
    assert "User: Tell me about the warranty." in prompt
    assert "Assistant: It lasts 24 months." in prompt
    assert "And the second one?" in prompt


def test_condensing_is_skipped_until_there_is_history() -> None:
    assert not needs_condensing([])
    assert not needs_condensing([MessageRecord(role="assistant", content="Hi")])
    assert needs_condensing([MessageRecord(role="user", content="Hi")])


def test_clean_condensed_strips_quotes_and_blank_replies() -> None:
    assert clean_condensed('"What is the price?"', "fallback") == "What is the price?"
    assert clean_condensed("   ", "fallback") == "fallback"
    assert clean_condensed("", "fallback") == "fallback"


def test_clean_condensed_takes_the_last_line_of_a_chatty_reply() -> None:
    raw = "Sure! Here is the rewritten question:\n\nWhat is the refund window?"
    assert clean_condensed(raw, "fallback") == "What is the refund window?"


def test_clean_condensed_rejects_a_runaway_rewrite() -> None:
    assert clean_condensed("word " * 500, "original question") == "original question"


def test_refusal_text_does_not_pretend_to_answer() -> None:
    assert "could not find" in NO_CONTEXT_ANSWER.lower()
