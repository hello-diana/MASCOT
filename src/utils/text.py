"""Consolidated response/text cleaning operations (single source of truth).

Before Phase 2 the repository carried several near-duplicate `<think>`-tag
cleaners whose edge behaviors intentionally differ. Each behavior is exposed
here as an explicit operation; the legacy entry points delegate to these so no
call site changes behavior:

- ``strip_closed_reasoning``: removes every CLOSED ``<think>...</think>``
  block, preserving surrounding text (legacy ``data_utils.clean_response``).
- ``extract_final_response``: splits on ``</think>`` and keeps what follows,
  dropping any text before the tag (legacy ``llm_as_a_judge.clean_response``,
  ``generate_and_annotate._clean``, the transcript formatters, and — with
  ``strip_without_tag=False`` — the rubric evaluators and
  ``parse_json_response``; with ``take_last_tag=True`` — the
  persona-template inline cleaner).
- ``strip_response_prefix``: removes an echoed "Your/JSON Response:" prefix.

The behaviors are pinned by tests/test_text_cleaning.py.
"""

import re
from typing import Optional

_CLOSED_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_RESPONSE_PREFIX_RE = re.compile(r"^(?:Your\s+)?(?:JSON\s+)?[Rr]esponse\s*:\s*")

THINK_CLOSE_TAG = "</think>"


def strip_closed_reasoning(text: Optional[str]) -> str:
    """Remove all closed ``<think>...</think>`` blocks and strip.

    Text before/between/after the blocks is preserved; an unclosed
    ``<think>`` block is left intact. ``None``/empty input returns ``''``.
    """
    if not text:
        return ""
    return _CLOSED_THINK_RE.sub("", text).strip()


def extract_final_response(text: Optional[str], *,
                           strip_without_tag: bool = True,
                           take_last_tag: bool = False) -> str:
    """Keep only the text after a ``</think>`` tag.

    Unlike :func:`strip_closed_reasoning`, any text BEFORE the tag is dropped.

    Args:
        text: Raw model output (``None``/empty returns ``''``).
        strip_without_tag: When no tag is present, whether to return the
            stripped input (default, most cleaners) or the original string
            verbatim (rubric evaluators, ``parse_json_response``).
        take_last_tag: Split on the LAST tag instead of the first when several
            closing tags are present (persona-template behavior).
    """
    if text is None:
        return ""
    if THINK_CLOSE_TAG in text:
        if take_last_tag:
            return text.rsplit(THINK_CLOSE_TAG, 1)[-1].strip()
        return text.split(THINK_CLOSE_TAG, 1)[1].strip()
    return text.strip() if strip_without_tag else text


def strip_response_prefix(text: Optional[str]) -> str:
    """Remove an echoed prompt prefix like ``'Your response:'`` from output."""
    if not text:
        return ""
    return _RESPONSE_PREFIX_RE.sub("", text.strip()).strip()
