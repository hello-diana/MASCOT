"""Frozen Director/Speaker/User rollout machinery for Stage-2 group training.

Moved verbatim from src/train/train_group.py (Phase-2 Part B): batched
directive generation with retry, dataset-dispatched prompts, prefix
construction, suffix rollout, and group-reward evaluation during rollout.
QMSum prompt routing stays byte-identical (the dispatchers route to the same
_qmsum template functions). This module must not import director_grpo or
director_dpo.
"""

import json
import logging
import random

import torch

from src.const import QMSUM_DATASET
from src.generation.mascot_episode import parse_directive
from src.generation.selection import fallback_directive
from src.models.llm_models import AgentResponse
from src.prompts.persona_templates import MascotPersonaTemplates
from src.prompts.prompt_reward_model import get_prompt_collective_reward_model
from src.rubrics import PAPER_V1
from src.user.simulated_users import (extract_user_json,
                                      generate_user_simulation_prompt,
                                      generate_user_simulation_prompt_qmsum)
from src.utils.data_utils import clean_response

from config.persona_config import AGENTS

logger = logging.getLogger(__name__)


# =====================================================================
# Batch directive generation with retry
# =====================================================================

def _random_agent(agents: dict) -> dict:
    """Create a random fallback directive (uniform strategy, see selection.py)."""
    return fallback_directive(agents, strategy="uniform").directive


def _generate_directives_batch(
    directive_generator,
    episode_states: list,
    agents: dict,
    indices: list,
    max_retries: int = 3,
    desc: str = "directives",
    dataset_name: str = QMSUM_DATASET,
) -> list:
    """Batch-generate and parse directives with retry on parse failure.

    Args:
        directive_generator: Frozen LLM with generate_batch().
        episode_states: List of episode state dicts (need meeting_context,
            conversation_history, agents_spoken).
        agents: Agent dictionary.
        indices: Which episode_states indices to generate for.
        max_retries: Max retries for failed directives.
        desc: Description for progress bar.

    Returns:
        List of parsed directive dicts, one per index in ``indices``.
    """
    if not indices:
        return []

    prompts = []
    for i in indices:
        ep = episode_states[i]
        prompts.append(MascotPersonaTemplates.directive_prompt(
            dataset_name,
            agents=agents,
            meeting_context=ep["meeting_context"],
            conversation_history=ep["conversation_history"],
            agents_spoken=ep["agents_spoken"],
        ))

    results = [None] * len(indices)
    pending = list(range(len(indices)))

    for attempt in range(1 + max_retries):
        if not pending:
            break

        batch_prompts = [prompts[j] for j in pending]
        label = desc if attempt == 0 else f"{desc} retry {attempt}"
        if attempt > 0:
            logger.debug(f"Retrying {len(pending)} failed directives (attempt {attempt})")
        raw_outputs = directive_generator.generate_batch(
            batch_prompts, max_tokens=1024, temperature=0.9, desc=label,
            system_prompt=None,
        )

        still_pending = []
        for jj, j in enumerate(pending):
            d = parse_directive(raw_outputs[jj], agents)
            if d is not None:
                results[j] = d
            else:
                still_pending.append(j)
        pending = still_pending

    # Last resort fallback after all retries exhausted
    if pending:
        logger.warning(
            f"{len(pending)} directives still unparseable after {max_retries} retries, "
            "using random agent fallback"
        )
        for j in pending:
            results[j] = _random_agent(agents)

    return results


# =====================================================================
# Prefix generation (turns 0..k-1)
# =====================================================================

def _user_sim_prompt(dataset_name: str, state: dict) -> str:
    """Dataset-dispatched user-simulation prompt for rollout states.

    Rollout states do not track an emotional trajectory, so the empathetic
    variant uses a neutral current emotion (documented simplification).
    """
    if dataset_name == QMSUM_DATASET:
        return generate_user_simulation_prompt_qmsum(
            state["meeting_context"], state["conversation_history"])
    return generate_user_simulation_prompt(
        initial_situation=state["meeting_context"],
        current_emotion="neutral",
        conversation_history=state["conversation_history"],
        user_big_five=None,
    )


def build_prefix_turns(
    directive_generator,
    agent_generator,
    meeting_contexts: list,
    k_values: list,
    agents: dict,
    user_simulator=None,
    user_response_probability: float = 0.3,
    dataset_name: str = QMSUM_DATASET,
) -> list:
    """Pre-generate prefix turns 0..k-1 for each sample using frozen vLLM.

    Processes turns sequentially; within each turn, generates directives
    and agent responses in batch (only for samples that need that turn).

    Args:
        directive_generator: Frozen LLM for directive generation.
        agent_generator: Frozen LLM for agent responses.
        meeting_contexts: Meeting context string per sample.
        k_values: Which director turn to train per sample (0 = no prefix).
        agents: Agent dictionary.
        user_simulator: Optional user simulation LLM.
        user_response_probability: Probability of user turn per round.

    Returns:
        List of state dicts with keys: conversation_history, agents_spoken,
        prefix_turns, meeting_context.
    """

    states = []
    for i in range(len(meeting_contexts)):
        states.append({
            "meeting_context": meeting_contexts[i],
            "conversation_history": "",
            "agents_spoken": {name: 0 for name in agents},
            "prefix_turns": [],
        })

    max_k = max(k_values) if k_values else 0

    for turn_idx in range(max_k):
        # Only process samples that need this prefix turn
        need_idx = [i for i, k in enumerate(k_values) if k > turn_idx]
        if not need_idx:
            break

        logger.info(f"Prefix turn {turn_idx}: generating for {len(need_idx)} samples")

        # 1. Batch generate directives (frozen director)
        directives = _generate_directives_batch(
            directive_generator, states, agents, need_idx,
            desc=f"Prefix turn {turn_idx} directives",
            dataset_name=dataset_name,
        )

        # 2. Batch generate agent responses (frozen speaker)
        agent_prompts = []
        for j, i in enumerate(need_idx):
            agent_name = directives[j]["agent_name"]
            persona_spec = agents[agent_name]
            agent_prompts.append(MascotPersonaTemplates.speaker_prompt(
                dataset_name,
                persona=persona_spec,
                context=states[i]["meeting_context"],
                meeting_summary=None,
                conversation_history=states[i]["conversation_history"],
            ))

        agent_responses = agent_generator.generate_batch(
            agent_prompts, max_tokens=512, temperature=0.9,
            system_prompt=None,
            desc=f"Prefix turn {turn_idx} responses",
        )

        # 3. Update states
        for j, i in enumerate(need_idx):
            agent_name = directives[j]["agent_name"]
            clean = clean_response(agent_responses[j])
            states[i]["conversation_history"] += f"[{agent_name}]: {clean}\n"
            states[i]["agents_spoken"][agent_name] += 1
            states[i]["prefix_turns"].append({
                "turn_idx": turn_idx,
                "directive": directives[j],
                "agent_name": agent_name,
                "response": agent_responses[j],
                "agent_response": agent_responses[j],
                "turn_type": "agent",
            })

        # 4. Optional user simulation
        if user_simulator is not None and random.random() < user_response_probability:
            user_prompts = [_user_sim_prompt(dataset_name, states[i]) for i in need_idx]
            user_responses = user_simulator.generate_batch(
                user_prompts, max_tokens=256, temperature=0.7,
                system_prompt=None,
                desc=f"Prefix turn {turn_idx} user sim",
            )
            for j, i in enumerate(need_idx):
                user_text = user_responses[j]
                parsed = extract_user_json(user_text)
                if parsed is not None:
                    user_text = parsed.get("response", user_text)
                clean_user = clean_response(user_text)
                states[i]["conversation_history"] += f"[User]: {clean_user}\n"
                states[i]["prefix_turns"].append({
                    "turn_idx": turn_idx,
                    "directive": None,
                    "agent_name": "User",
                    "response": user_text,
                    "agent_response": user_text,
                    "turn_type": "user",
                })

    logger.info(
        f"Built prefixes for {len(meeting_contexts)} samples "
        f"(k range: {min(k_values)}-{max(k_values) if k_values else 0})"
    )
    return states


# =====================================================================
# Suffix rollout (turns k..k+num_suffix-1)
# =====================================================================

def rollout_suffix_turns(
    turn_k_directives: list,
    k_values: list,
    prefix_states: list,
    directive_generator,
    agent_generator,
    meeting_contexts: list,
    agents: dict,
    num_episode_turns: int = 5,
    user_simulator=None,
    user_response_probability: float = 0.3,
    dataset_name: str = QMSUM_DATASET,
) -> list:
    """Roll out suffix turns starting from the training turn.

    Generates exactly ``num_episode_turns - k`` suffix rounds so that each
    episode has ``num_episode_turns`` total turns (prefix + suffix).
    Round 0 (suffix_idx=0) uses GRPO-generated directives (training turn);
    subsequent rounds use the frozen directive_generator.

    Args:
        turn_k_directives: Parsed directives from GRPO completions.
        k_values: Which turn each sample trained on.
        prefix_states: States from build_prefix_turns().
        directive_generator: Frozen LLM for suffix directive generation.
        agent_generator: Frozen LLM for agent responses.
        meeting_contexts: Meeting context per sample.
        agents: Agent dictionary.
        num_episode_turns: Total episode length (prefix + suffix).
        user_simulator: Optional user simulation LLM.
        user_response_probability: Probability of user turn per round.

    Returns:
        List of episode dicts with turns, conversation_history, and agents_spoken.
    """
    batch_size = len(turn_k_directives)

    # Initialize from prefix states
    episodes = []
    for i in range(batch_size):
        episodes.append({
            "meeting_context": meeting_contexts[i],
            "turns": list(prefix_states[i]["prefix_turns"]),
            "conversation_history": prefix_states[i]["conversation_history"],
            "agents_spoken": dict(prefix_states[i]["agents_spoken"]),
        })

    all_idx = list(range(batch_size))

    # Compute exact suffix turns needed. Batches are constructed k-homogeneous
    # upstream (mixed-k remainders are dropped); a mixed batch here means the
    # batching invariant was violated, so fail loudly rather than roll out
    # wrong-length episodes.
    if len(set(k_values)) > 1:
        raise ValueError(
            f"rollout_suffix_turns received a mixed-k batch (k values: "
            f"{sorted(set(k_values))}); batches must be k-homogeneous")
    k = k_values[0]
    num_suffix = num_episode_turns - k
    if num_suffix <= 0:
        logger.warning(f"k={k} >= num_episode_turns={num_episode_turns}, no suffix turns to generate")
        return episodes

    for suffix_idx in range(num_suffix):
        actual_turn_indices = [k_values[i] + suffix_idx for i in range(batch_size)]

        # --- Determine directives ---
        if suffix_idx == 0:
            # Training turn: use GRPO-generated directives
            directives = list(turn_k_directives)
            # Retry any that failed to parse with frozen director
            retry_needed = [i for i in range(batch_size) if directives[i] is None]
            if retry_needed:
                logger.debug(
                    f"Retrying {len(retry_needed)} unparsed GRPO directives "
                    "with frozen director"
                )
                retry_results = _generate_directives_batch(
                    directive_generator, episodes, agents, retry_needed,
                    desc="GRPO directive retry",
                    dataset_name=dataset_name,
                )
                for j, i in enumerate(retry_needed):
                    directives[i] = retry_results[j]
        else:
            # Frozen suffix turns: batch generate directives
            dir_results = _generate_directives_batch(
                directive_generator, episodes, agents, all_idx,
                desc=f"Suffix turn {suffix_idx} directives",
                dataset_name=dataset_name,
            )
            directives = dir_results

        # --- Batch generate agent responses ---
        agent_prompts = []
        for i in all_idx:
            agent_name = directives[i]["agent_name"]
            persona_spec = agents[agent_name]
            agent_prompts.append(MascotPersonaTemplates.speaker_prompt(
                dataset_name,
                persona=persona_spec,
                context=episodes[i]["meeting_context"],
                meeting_summary=None,
                conversation_history=episodes[i]["conversation_history"],
            ))

        agent_responses = agent_generator.generate_batch(
            agent_prompts, max_tokens=4096, temperature=0.9,
            response_format=AgentResponse,
            system_prompt=None,
            desc=f"Suffix turn {suffix_idx} responses",
        )

        # --- Update episode states ---
        for i in all_idx:
            agent_name = directives[i]["agent_name"]
            raw = agent_responses[i]
            # Parse structured output; fall back to raw text
            try:
                parsed = json.loads(raw)
                response_text = parsed["response"]
            except (json.JSONDecodeError, KeyError, TypeError):
                response_text = raw
            clean = clean_response(response_text)
            episodes[i]["conversation_history"] += f"[{agent_name}]: {clean}\n"
            episodes[i]["agents_spoken"][agent_name] = \
                episodes[i]["agents_spoken"].get(agent_name, 0) + 1
            episodes[i]["turns"].append({
                "turn_idx": actual_turn_indices[i],
                "directive": directives[i],
                "agent_name": agent_name,
                "response": response_text,
                "agent_response": response_text,
                "turn_type": "agent",
            })

        # --- Optional user simulation ---
        if user_simulator is not None and random.random() < user_response_probability:
            user_prompts = [_user_sim_prompt(dataset_name, episodes[i]) for i in all_idx]
            user_responses = user_simulator.generate_batch(
                user_prompts, max_tokens=256, temperature=0.7,
                system_prompt=None,
                desc=f"Suffix turn {suffix_idx} user sim",
            )
            for i in all_idx:
                user_text = user_responses[i]
                parsed_user = extract_user_json(user_text)
                if parsed_user is not None:
                    user_text = parsed_user.get("response", user_text)
                clean_user = clean_response(user_text)
                episodes[i]["conversation_history"] += f"[User]: {clean_user}\n"
                episodes[i]["turns"].append({
                    "turn_idx": actual_turn_indices[i],
                    "directive": None,
                    "agent_name": "User",
                    "response": user_text,
                    "agent_response": user_text,
                    "turn_type": "user",
                })

    logger.info(
        f"Rolled out {batch_size} episodes x {num_suffix} suffix turns "
        f"(k={k}, total={num_episode_turns} turns per episode)"
    )
    return episodes


# =====================================================================
# Reward computation
# =====================================================================

def compute_group_reward(
    prompts,
    completions,
    meeting_context=None,
    train_k=None,
    prefix_conversation_history=None,
    prefix_agents_spoken=None,
    prefix_turns=None,
    directive_generator=None,
    agents_vllm=None,
    agents_dict=None,
    group_reward_model=None,
    group_reward_tokenizer=None,
    user_simulator=None,
    user_response_probability=0.3,
    num_episode_turns=5,
    rubric_version=None,
    dataset_name=QMSUM_DATASET,
    **kwargs,
):
    """Compute group-level rewards for directive completions.

    Flow:
        1. Parse turn-k directives from GRPO ``completions``.
        2. Reconstruct prefix states from dataset columns (passed by TRL).
        3. Roll out suffix turns (training turn + frozen continuation).
        4. Truncate to first ``num_episode_turns`` rounds.
        5. Score with the group reward model.

    Extra dataset columns (train_k, prefix_conversation_history,
    prefix_agents_spoken, prefix_turns) are passed by TRL GRPOTrainer
    as keyword arguments.

    Returns:
        list[float]: Rewards, one per completion.
    """
    if agents_dict is None:
        agents_dict = AGENTS

    batch_size = len(completions)
    logger.debug(f"Computing group rewards for batch of {batch_size}")

    # --- Parse k values (TRL may pass as list[int] or tensor) ---
    if train_k is not None:
        k_list = [int(k) for k in train_k]
    else:
        k_list = [0] * batch_size

    # --- Reconstruct prefix states from stored dataset columns ---
    prefix_states_list = []
    for i in range(batch_size):
        conv_hist = (
            prefix_conversation_history[i]
            if prefix_conversation_history
            else ""
        )
        agents_spoken = (
            json.loads(prefix_agents_spoken[i])
            if prefix_agents_spoken
            else {name: 0 for name in agents_dict}
        )
        p_turns = (
            json.loads(prefix_turns[i])
            if prefix_turns
            else []
        )
        mc = meeting_context[i] if meeting_context else ""
        prefix_states_list.append({
            "conversation_history": conv_hist,
            "agents_spoken": agents_spoken,
            "prefix_turns": p_turns,
            "meeting_context": mc,
        })

    # --- Parse GRPO completions as turn-k directives ---
    turn_k_directives = []
    for comp in completions:
        if isinstance(comp, list) and len(comp) > 0:
            directive_text = comp[0].get('content', '')
        else:
            directive_text = str(comp)
        d = parse_directive(directive_text, agents_dict)
        turn_k_directives.append(d)  # may be None; rollout handles retry

    # --- Rollout suffix turns ---
    mc_list = (
        meeting_context if meeting_context else [""] * batch_size
    )
    try:
        episodes = rollout_suffix_turns(
            turn_k_directives=turn_k_directives,
            k_values=k_list,
            prefix_states=prefix_states_list,
            directive_generator=directive_generator,
            agent_generator=agents_vllm,
            meeting_contexts=mc_list,
            agents=agents_dict,
            num_episode_turns=num_episode_turns,
            user_simulator=user_simulator,
            user_response_probability=user_response_probability,
            dataset_name=dataset_name,
        )
    except Exception as e:
        logger.error(f"Suffix rollout failed: {e}", exc_info=True)
        return [0.0] * batch_size

    # --- Format and score ---
    episode_texts = [get_prompt_collective_reward_model(ep["turns"], rubric_version=rubric_version or PAPER_V1)
                     for ep in episodes]

    inputs = group_reward_tokenizer(
        episode_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(group_reward_model.device)

    with torch.no_grad():
        outputs = group_reward_model(**inputs)
        logits = outputs.logits
        if logits.size(-1) == 2:
            rewards = logits[:, 0]
        else:
            rewards = logits.squeeze(-1)

    reward_list = rewards.cpu().tolist()
    logger.debug(f"Group rewards: mean={sum(reward_list)/len(reward_list):.3f}")
    return reward_list
