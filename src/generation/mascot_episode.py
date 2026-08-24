"""Shared multi-turn MASCOT episode generation loop."""

import json
import re
import random
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any
from collections import OrderedDict

from src.models.llm_models import AgentResponse
from src.generation.selection import (PERSONA_DEFAULT_DIRECTIVE,
                                      SelectionResult, fallback_directive)
from src.utils.run_metadata import record_failure

logger = logging.getLogger(__name__)


def _try_raw_decode(text: str):
    """Use json.JSONDecoder to parse the first complete JSON object in text.

    Tolerates trailing junk (extra blocks, prose, repeated objects) — which is
    exactly the "Extra data" failure mode we saw with Qwen3-8B directive output.
    Returns dict/list or None.
    """
    decoder = json.JSONDecoder()
    for start in range(len(text)):
        if text[start] not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
            return obj
        except json.JSONDecodeError:
            continue
    return None


def parse_directive(response: str, agents: dict) -> Optional[Dict]:
    """Parse directive from model output with structured extraction and fallback.

    Strategy (each tried in order, returns first valid result):
    1. Extract content from ```json ... ``` markdown blocks and parse.
    2. Use json.JSONDecoder.raw_decode to read the first valid JSON object
       (tolerates "Extra data" / repeated blocks).
    3. Fuzzy: pick first valid agent_name from text and synthesize a directive.
    4. Return None (caller does random fallback).
    """

    def _validate(parsed):
        if isinstance(parsed, list) and len(parsed) > 0:
            parsed = parsed[0]
        if not isinstance(parsed, dict):
            return None
        if parsed.get("agent_name") not in agents:
            return None
        # action / tone may be missing — fill defaults rather than reject
        return {
            "agent_name": parsed["agent_name"],
            "action": parsed.get("action") or "Add your unique perspective",
            "tone": parsed.get("tone") or "use your best judgment",
        }

    # 1. Markdown code block(s) — try each one
    for m in re.finditer(r'```(?:json)?\s*([\s\S]*?)\s*```', response):
        json_str = m.group(1).strip()
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            parsed = _try_raw_decode(json_str)
        result = _validate(parsed) if parsed is not None else None
        if result is not None:
            return result

    # 2. Raw scan over the whole response (handles missing/incorrect code fences)
    parsed = _try_raw_decode(response)
    result = _validate(parsed) if parsed is not None else None
    if result is not None:
        return result

    # 3. Fuzzy: any agent name in the response wins
    for name in agents:
        if re.search(r'\b' + re.escape(name) + r'\b', response):
            return {
                "agent_name": name,
                "action": "Add your unique perspective",
                "tone": "use your best judgment",
            }

    logger.debug(f"Invalid response JSON (no agent_name found):\n{response[:300]}")
    return None


@dataclass
class EpisodeConfig:
    """Configuration for multi-turn MASCOT episode generation."""
    agents: dict
    num_turns: int = 10
    directive_temperature: float = 0.9
    agent_temperature: float = 0.9
    directive_max_tokens: int = 4096
    agent_max_tokens: int = 4096
    enable_thinking: bool = True
    disable_directive: bool = False
    # When True, the agent model is generated under a JSON schema that forces
    # ``{"response": "..."}`` output (matches train_group.py rollout). Needed
    # for LoRA checkpoints trained with structured-output rollouts — without it
    # the model can loop and emit duplicated replies.
    structured_agent_output: bool = True
    # Strict mode: any generation failure, malformed directive, or empty
    # response terminates the run instead of being counted and tolerated.
    strict: bool = False


# Single source of truth for fallback selection lives in
# src/generation/selection.py; this alias keeps the historical name.
_PERSONA_DEFAULT_DIRECTIVE = PERSONA_DEFAULT_DIRECTIVE


def _random_agent(agents: dict, agents_spoken: dict) -> dict:
    """Pick an agent, prioritizing those who haven't spoken yet.

    When directive parsing fails, we still want a meaningful (action, tone)
    pair so the agent doesn't fall back to its most common SFT phrase.
    """
    return fallback_directive(agents, strategy="least_spoken",
                              agents_spoken=agents_spoken).directive


def run_mascot_episode_loop(
    conversation_states: OrderedDict,
    directive_generator: Any,
    agent_generator: Any,
    config: EpisodeConfig,
    build_directive_prompt: Callable[[dict], str],
    build_agent_prompt: Callable[[dict, dict], str],
    on_turn_end: Optional[Callable[[OrderedDict, int], None]] = None,
    counters: Optional[Dict[str, int]] = None,
) -> OrderedDict:
    """Run the shared MASCOT multi-turn generation loop.

    Args:
        conversation_states: OrderedDict of key -> state dict. Each state must
            contain ``"turns": []`` and ``"agents_spoken": {name: 0, ...}``.
        directive_generator: LLM with ``generate_batch()`` method.
        agent_generator: LLM with ``generate_batch()`` method.
        config: Generation hyperparameters.
        build_directive_prompt: ``state -> prompt_str`` for directive generation.
        build_agent_prompt: ``(state, directive_dict) -> prompt_str`` for agent
            response generation.
        on_turn_end: Optional hook called after each turn with
            ``(conversation_states, turn_idx)``.  Used by empathetic eval to run
            user simulation.
        counters: Optional failure-counter dict (see
            ``src.utils.run_metadata.new_counters``), mutated in place. In
            ``config.strict`` mode the first failure raises instead.

    Returns:
        The same *conversation_states* OrderedDict, mutated in-place.
    """
    agents = config.agents
    keys = list(conversation_states.keys())

    for turn_idx in range(config.num_turns):
        n = len(keys)
        logger.info(f"=== Turn {turn_idx + 1}/{config.num_turns}  ({n} conversations) ===")

        # ---- Directive phase ------------------------------------------------
        if config.disable_directive:
            directive_outputs = ["[RANDOM]"] * n
            directives = [_random_agent(agents, conversation_states[k]["agents_spoken"]) for k in keys]
        else:
            directive_prompts = [build_directive_prompt(conversation_states[k]) for k in keys]
            directive_outputs = directive_generator.generate_batch(
                prompts=directive_prompts,
                max_tokens=config.directive_max_tokens,
                temperature=config.directive_temperature,
                enable_thinking=config.enable_thinking,
                system_prompt=None,  # prompts are self-contained
                desc=f"Directives turn {turn_idx+1}",
            )
            directives = []
            fallback = 0
            for i, output in enumerate(directive_outputs):
                if not (output or "").strip():
                    # Exhausted-retry generation failure surfaces as "".
                    record_failure(counters, "num_generation_failures", config.strict,
                                   f"empty directive output (turn {turn_idx}, conv {keys[i]})")
                d = parse_directive(output, agents)
                if d is None:
                    record_failure(counters, "num_directive_parse_failures", config.strict,
                                   f"unparseable directive (turn {turn_idx}, conv {keys[i]})")
                    d = _random_agent(agents, conversation_states[keys[i]]["agents_spoken"])
                    record_failure(counters, "num_random_agent_fallbacks", False)
                    fallback += 1
                directives.append(d)
            if fallback:
                logger.info(f"  Directive fallback: {fallback}/{n}")

        # ---- Agent response phase -------------------------------------------
        agent_prompts = [
            build_agent_prompt(conversation_states[keys[i]], directives[i])
            for i in range(n)
        ]
        gen_kwargs = dict(
            prompts=agent_prompts,
            max_tokens=config.agent_max_tokens,
            temperature=config.agent_temperature,
            enable_thinking=config.enable_thinking,
            system_prompt=None,  # prompts are self-contained
            desc=f"Agent responses turn {turn_idx+1}",
        )
        if config.structured_agent_output:
            gen_kwargs["response_format"] = AgentResponse
        agent_responses = agent_generator.generate_batch(**gen_kwargs)

        # ---- Update states --------------------------------------------------
        for i, k in enumerate(keys):
            state = conversation_states[k]
            name = directives[i]["agent_name"]
            state["agents_spoken"][name] += 1
            raw = agent_responses[i]
            if config.structured_agent_output:
                try:
                    response_text = json.loads(raw)["response"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    response_text = raw
            else:
                response_text = raw
            if not (response_text or "").strip():
                record_failure(counters, "num_empty_responses", config.strict,
                               f"empty agent response (turn {turn_idx}, conv {k})")
            state["turns"].append({
                "turn_idx": turn_idx,
                # One round = one Director->Speaker(->User) cycle. Used by
                # metadata-driven pairing in automatic metrics.
                "round_id": turn_idx,
                "directive": directives[i],
                "agent_name": name,
                "response": response_text,
                "directive_output": directive_outputs[i],
            })

        # ---- Post-turn hook (e.g. user simulation) --------------------------
        if on_turn_end is not None:
            on_turn_end(conversation_states, turn_idx)

    logger.info(f"Generated {len(conversation_states)} complete conversations")
    return conversation_states
