"""Prompt construction — pure string functions, no I/O.

Keeping this module free of config and network calls is what lets the tests
assert on exact prompt text without an API key.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.db.models import MessageRecord, SearchHit

ANSWER_SYSTEM_PROMPT = """\
You are a retrieval-grounded assistant. You answer strictly from the numbered \
context passages supplied with each question.

Rules:
1. Use ONLY the context passages. Never use outside knowledge, and never guess.
2. Cite the passage number in square brackets immediately after each claim it \
supports, like [1] or [2][4]. Every factual sentence must carry a citation.
3. If the passages do not contain the answer, say so plainly and stop. Do not \
speculate, and do not offer a partial answer dressed up as a complete one.
4. If the passages disagree with each other, say so and cite both sides.
5. Answer in the user's language, in prose. Be concise and concrete; do not pad \
the answer or restate the question.
"""

CONDENSE_SYSTEM_PROMPT = """\
You rewrite a follow-up question into a standalone one.

Given a conversation and the latest user message, produce a single question that \
can be understood with no prior context: resolve every pronoun and implicit \
reference using the conversation.

Output ONLY the rewritten question. No preamble, no quotes, no explanation. If \
the latest message already stands alone, output it unchanged.
"""

# Returned without calling the LLM when retrieval finds nothing above the
# similarity threshold — the cheapest and most reliable anti-hallucination step.
NO_CONTEXT_ANSWER = (
    "I could not find anything about that in your uploaded documents, so I "
    "cannot answer it. Try rephrasing the question using wording closer to the "
    "documents, or upload a source that covers this topic."
)


def format_context(
    hits: Sequence[SearchHit], *, max_chars: int
) -> tuple[str, list[SearchHit]]:
    """Render hits as numbered passages, stopping before `max_chars`.

    Returns the rendered context and the hits actually included, so citations
    shown to the user always match the passages the model could see.
    """
    blocks: list[str] = []
    used: list[SearchHit] = []
    total = 0

    for hit in hits:
        block = (
            f"[{len(used) + 1}] Source: {hit.filename}"
            f"{f', page {hit.page}' if hit.page is not None else ''}\n"
            f"{hit.content}"
        )
        # Always admit the first passage; otherwise an unusually long top hit
        # would produce an empty context and a spurious "not found".
        if used and total + len(block) > max_chars:
            break
        blocks.append(block)
        used.append(hit)
        total += len(block)

    return "\n\n".join(blocks), used


def build_answer_prompt(question: str, context: str) -> str:
    return (
        f"Context passages:\n\n{context}\n\n"
        f"---\n"
        f"Question: {question.strip()}\n\n"
        f"Answer using only the passages above, citing them by number."
    )


def build_condense_prompt(history: Sequence[MessageRecord], question: str) -> str:
    transcript = "\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content.strip()}"
        for m in history
    )
    return (
        f"Conversation so far:\n{transcript}\n\n"
        f"Latest user message: {question.strip()}\n\n"
        f"Standalone question:"
    )


def needs_condensing(history: Sequence[MessageRecord]) -> bool:
    """Only worth an extra LLM call once there is something to resolve against."""
    return any(m.role == "user" for m in history)


def clean_condensed(raw: str, fallback: str) -> str:
    """Guard against a model that ignores 'output only the question'."""
    text = (raw or "").strip().strip('"').strip()
    if not text:
        return fallback
    # A multi-line reply usually means it added commentary; keep the last
    # non-empty line, which is where the rewritten question ends up.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = lines[-1] if lines else fallback
    # A "rewrite" several times longer than the original is a runaway answer.
    if len(text) > max(400, len(fallback) * 6):
        return fallback
    return text
