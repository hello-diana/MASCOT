"""Shared Director-selection fallback logic (single source of truth).

Before Phase 2 the episode generator (src/generation/mascot_episode.py) and
the group trainer (src/train/train_group.py) each carried their own
``_random_agent`` fallback with intentionally different behavior. Both are
preserved here as explicit strategies:

- ``strategy="least_spoken"`` (episode loop): prefer agents that have not
  spoken yet (uniform among them; else uniform among all) and use a
  persona-specific default (action, tone) so the Speaker still gets a
  meaningful directive.
- ``strategy="uniform"`` (training rollout): uniform choice over all agents
  with a generic (action, tone).

The RNG is an explicit argument; it defaults to the global ``random`` module
so the pre-refactor random sequences under a given seed are unchanged.
"""

import random as _global_random
from dataclasses import dataclass
from typing import Dict, Optional

# Persona-specific default directives used by the least_spoken strategy so a
# fallback agent doesn't collapse to its most common SFT phrase.
PERSONA_DEFAULT_DIRECTIVE = {
    "the_anchor":   ("Validate the user's emotion with reflective, mirroring language. Do NOT advise or fix.",
                     "warm and patient"),
    "the_catalyst": ("Ask one Socratic, open-ended question that reframes the situation.",
                     "curious and analytical"),
    "the_beacon":   ("Identify a core human strength the user is showing and amplify it gently.",
                     "warm and encouraging"),
}

_GENERIC_ACTION_TONE = ("Add your unique perspective", "use your best judgment")
_UNIFORM_ACTION_TONE = ("Continue the ongoing discussion", "neutral")


@dataclass
class SelectionResult:
    """Outcome of a Director-selection step (parsed directive or fallback)."""
    agent_id: str
    directive: Dict[str, str]
    used_fallback: bool
    fallback_reason: Optional[str] = None


def fallback_directive(agents: dict, *, strategy: str,
                       agents_spoken: Optional[Dict[str, int]] = None,
                       fallback_reason: Optional[str] = None,
                       rng=_global_random) -> SelectionResult:
    """Build a fallback directive when directive parsing failed.

    Args:
        agents: Agent registry (name -> spec).
        strategy: ``"least_spoken"`` (episode loop) or ``"uniform"``
            (training rollout) — see module docstring; the strategies are
            intentionally distinct and must not be collapsed.
        agents_spoken: Per-agent turn counts (required for least_spoken).
        fallback_reason: Recorded on the result for reporting.
        rng: Random source (``random`` module or ``random.Random`` instance).
            Defaults to the global module to preserve seeded behavior.
    """
    if strategy == "least_spoken":
        if agents_spoken is None:
            raise ValueError("least_spoken strategy requires agents_spoken")
        unspoken = [n for n, c in agents_spoken.items() if c == 0]
        name = rng.choice(unspoken) if unspoken else rng.choice(list(agents.keys()))
        action, tone = PERSONA_DEFAULT_DIRECTIVE.get(name, _GENERIC_ACTION_TONE)
    elif strategy == "uniform":
        name = rng.choice(list(agents.keys()))
        action, tone = _UNIFORM_ACTION_TONE
    else:
        raise ValueError(f"Unknown selection fallback strategy: {strategy!r}")

    return SelectionResult(
        agent_id=name,
        directive={"agent_name": name, "action": action, "tone": tone},
        used_fallback=True,
        fallback_reason=fallback_reason,
    )
