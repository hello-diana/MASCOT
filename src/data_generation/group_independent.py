"""Independent group-episode generation pipeline + the per-dataset episode
hooks (directive/agent prompt builders and user-simulation turn hooks) shared
with the branching pipeline.

Moved verbatim from src/dataset/generate_and_annotate.py (Phase-2 Part A).
"""

import logging
import os
import random
from collections import OrderedDict
from typing import Dict, List

from datasets import load_dataset

from config.persona_config import AGENTS, EMPATHETIC_AGENTS
from src.const import (DATASET2HF, EMPATHETIC_DATASET, ESCONV_DATASET,
                       QMSUM_DATASET)
from src.data_generation.annotation import (annotate_group_episodes,
                                            verify_pairs_with_llm)
from src.data_generation.common import artifact_paths, scale_scores
from src.data_generation.preference_pairs import build_independent_group_pairs
from src.data_generation.serialization import (read_jsonl, save_pairs_dataset,
                                               write_jsonl)
from src.eval.rubric_evaluator import (GroupRubricEvaluator,
                                       IndividualRubricEvaluator)
from src.generation.mascot_episode import EpisodeConfig, run_mascot_episode_loop
from src.models.llm_models import VLLMServerModel
from src.prompts.persona_templates import MascotPersonaTemplates
from src.user.simulated_users import (extract_user_json,
                                      generate_user_simulation_prompt,
                                      generate_user_simulation_prompt_qmsum)
from src.utils.data_utils import (load_empathetic_situations,
                                  load_esconv_situations)
from src.utils.transcript_utils import (create_meeting_contexts,
                                        format_conversation_history,
                                        format_conversation_history_empathetic)

logger = logging.getLogger(__name__)


# =====================================================================
# Shared helpers for group pipelines (QMSum)
# =====================================================================

def _build_directive_qmsum(state: dict, agents: dict) -> str:
    """Build directive prompt for QMSum group episodes."""
    h = format_conversation_history(state["turns"], name_key='agent_name')
    return MascotPersonaTemplates.directive_prompt_qmsum(
        agents=agents, meeting_context=state["meeting"]["meeting_context"],
        conversation_history=h, agents_spoken=state["agents_spoken"],
    )


def _build_agent_qmsum(state: dict, directive: dict, agents: dict) -> str:
    """Build agent speaking prompt for QMSum group episodes."""
    h = format_conversation_history(state["turns"], name_key='agent_name')
    return MascotPersonaTemplates.speaker_prompt_qmsum(
        persona=agents[directive["agent_name"]],
        context=state["meeting"]["meeting_context"],
        meeting_summary=None, conversation_history=h,
    )


def _on_turn_end_qmsum(
    conv_states: OrderedDict, turn_idx: int,
    llm: VLLMServerModel, user_response_probability: float,
) -> None:
    """Post-turn hook: tag turn_type and optionally run QMSum user simulation."""
    for state in conv_states.values():
        last = state["turns"][-1]
        if "turn_type" not in last:
            last["turn_type"] = "agent"

    if user_response_probability <= 0:
        return

    user_keys = []
    user_prompts = []
    for key, state in conv_states.items():
        if random.random() < user_response_probability:
            user_keys.append(key)
            h = format_conversation_history(state["turns"], name_key='agent_name')
            user_prompts.append(generate_user_simulation_prompt_qmsum(
                meeting_context=state["meeting"]["meeting_context"],
                conversation_history=h,
            ))

    if not user_keys:
        return

    user_responses = [(t or "").strip() for t in llm.generate_batch(
        prompts=user_prompts, max_tokens=512,
        temperature=0.7,
        system_prompt=None,
        desc=f"User sim QMSum turn {turn_idx} ({len(user_prompts)})",
    )]

    for i, key in enumerate(user_keys):
        state = conv_states[key]
        raw = user_responses[i].strip()
        if not raw:
            continue
        parsed = extract_user_json(raw)
        if parsed is not None:
            user_text = parsed.get("response")
            if user_text:
                state["turns"].append({
                    "turn_idx": turn_idx, "round_id": turn_idx,
                    "turn_type": "user",
                    "response": user_text,
                    "agent_name": "user",
                })


# =====================================================================
# Shared helpers for group pipelines (Empathetic)
# =====================================================================

def _build_directive_emp(state: dict, agents: dict) -> str:
    """Build directive prompt for Empathetic group episodes."""
    h = format_conversation_history_empathetic(state["turns"])
    return MascotPersonaTemplates.directive_prompt_empathetic(
        agents=agents, situation=state['initial_situation'],
        emotion=state.get('current_emotion', ''),
        conversation_history=h, agents_spoken=state["agents_spoken"],
    )


def _build_agent_emp(state: dict, directive: dict, agents: dict) -> str:
    """Build agent speaking prompt for Empathetic group episodes."""
    h = format_conversation_history_empathetic(state["turns"])
    return MascotPersonaTemplates.speaker_prompt_empathetic(
        persona=agents[directive["agent_name"]],
        context=state['initial_situation'],
        conversation_history=h, num_shot=5,
    )


def _on_turn_end_emp(
    conv_states: OrderedDict, turn_idx: int,
    llm: VLLMServerModel, user_response_probability: float,
) -> None:
    """Post-turn hook: tag turn_type and optionally run Empathetic user simulation."""
    for state in conv_states.values():
        last = state["turns"][-1]
        if "turn_type" not in last:
            last["turn_type"] = "agent"

    if user_response_probability <= 0:
        return

    user_keys = []
    user_prompts = []
    for key, state in conv_states.items():
        if random.random() < user_response_probability:
            user_keys.append(key)
            h = format_conversation_history_empathetic(state["turns"])
            user_prompts.append(generate_user_simulation_prompt(
                initial_situation=state['initial_situation'],
                current_emotion=state.get('current_emotion', ''),
                conversation_history=h,
            ))

    if not user_keys:
        return

    user_responses = [(t or "").strip() for t in llm.generate_batch(
        prompts=user_prompts, max_tokens=512,
        temperature=0.7,
        system_prompt=None,
        desc=f"User sim Empathetic turn {turn_idx} ({len(user_prompts)})",
    )]

    for i, key in enumerate(user_keys):
        state = conv_states[key]
        raw = user_responses[i].strip()
        if not raw:
            continue
        parsed = extract_user_json(raw)
        if parsed is not None:
            user_text = parsed.get("response")
            new_emotion = parsed.get("new_emotion")
            if user_text:
                state["turns"].append({
                    "turn_idx": turn_idx, "round_id": turn_idx,
                    "turn_type": "user",
                    "response": user_text,
                    "agent_name": "user",
                })
                if new_emotion:
                    state["current_emotion"] = new_emotion


# =====================================================================
# GROUP PIPELINE (independent episodes)
# =====================================================================

def group_conversation_pipeline_independent(args, llm: VLLMServerModel, judge_llm: VLLMServerModel):
    """Generate N episodes per query via MASCOT loop, score at episode + turn level."""
    agents = EMPATHETIC_AGENTS if args.dataset in (EMPATHETIC_DATASET, ESCONV_DATASET) else AGENTS
    id_key = "meeting_id" if args.dataset == QMSUM_DATASET else "conv_id"
    adapted = llm

    gen_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(gen_dir, exist_ok=True)

    # ---- Phase 1: Generate N episodes per query ----
    _paths = artifact_paths(args)
    gen_file = _paths["raw_file"]

    if os.path.exists(gen_file):
        logger.info(f"Found existing group episodes at {gen_file}, skipping generation")
        all_episode_states: Dict[str, List[dict]] = {
            rec["query_key"]: rec["episodes"] for rec in read_jsonl(gen_file)
        }
    else:
        all_episode_states: Dict[str, List[dict]] = {}  # query_key -> list of episode states

        if args.dataset == QMSUM_DATASET:
            dataset = load_dataset(DATASET2HF[QMSUM_DATASET], split="train")
            if args.max_samples:
                dataset = dataset.select(range(min(args.max_samples, len(dataset))))
            meetings = create_meeting_contexts(dataset, args.turns_per_round)

            for ep_idx in range(args.num_episodes):
                logger.info(f"=== Episode {ep_idx + 1}/{args.num_episodes} ===")
                conversation_states = OrderedDict()
                for m in meetings:
                    conversation_states[(m['id'], m['start_idx'])] = {
                        "meeting": m, "id": m['id'], "start_idx": m['start_idx'],
                        "turns": [],
                        "agents_spoken": {n: 0 for n in agents},
                    }

                ep_config = EpisodeConfig(
                    agents=agents,
                    num_turns=args.num_turns,
                    directive_temperature=0.9,
                    agent_temperature=0.9,
                    directive_max_tokens=4096,
                    agent_max_tokens=4096,
                    enable_thinking=True,
                    disable_directive=False,
                )

                conversation_states = run_mascot_episode_loop(
                    conversation_states=conversation_states,
                    directive_generator=adapted,
                    agent_generator=adapted,
                    config=ep_config,
                    build_directive_prompt=lambda s: _build_directive_qmsum(s, agents),
                    build_agent_prompt=lambda s, d: _build_agent_qmsum(s, d, agents),
                    on_turn_end=lambda cs, ti: _on_turn_end_qmsum(
                        cs, ti, llm, args.user_response_probability),
                )

                for key, state in conversation_states.items():
                    qk = f"{key[0]}_{key[1]}"
                    all_episode_states.setdefault(qk, []).append(dict(state))

        elif args.dataset in (EMPATHETIC_DATASET, ESCONV_DATASET):
            if args.dataset == ESCONV_DATASET:
                conv_list = load_esconv_situations(split="train", max_samples=args.max_samples)
            else:
                conv_list = load_empathetic_situations(split="train", max_samples=args.max_samples)

            for ep_idx in range(args.num_episodes):
                logger.info(f"=== Episode {ep_idx + 1}/{args.num_episodes} ===")
                conversation_states = OrderedDict()
                for c in conv_list:
                    conversation_states[(c['conv_id'], None)] = {
                        "conv_id": c['conv_id'],
                        "initial_emotion": c['emotion'],
                        "initial_situation": c['situation'].replace("__comma__", ", "),
                        "current_emotion": c['emotion'],
                        "turns": [],
                        "agents_spoken": {n: 0 for n in agents},
                    }

                ep_config = EpisodeConfig(
                    agents=agents,
                    num_turns=args.num_turns,
                    directive_temperature=0.9,
                    agent_temperature=0.9,
                    directive_max_tokens=4096,
                    agent_max_tokens=4096,
                    enable_thinking=True,
                    disable_directive=False,
                )

                conversation_states = run_mascot_episode_loop(
                    conversation_states=conversation_states,
                    directive_generator=adapted,
                    agent_generator=adapted,
                    config=ep_config,
                    build_directive_prompt=lambda s: _build_directive_emp(s, agents),
                    build_agent_prompt=lambda s, d: _build_agent_emp(s, d, agents),
                    on_turn_end=lambda cs, ti: _on_turn_end_emp(
                        cs, ti, llm, args.user_response_probability),
                )

                for key, state in conversation_states.items():
                    qk = key[0]  # conv_id
                    all_episode_states.setdefault(qk, []).append(dict(state))

        # Save raw episodes
        write_jsonl(gen_file,
                    ({"query_key": qk, "episodes": episodes}
                     for qk, episodes in all_episode_states.items()))
        logger.info(f"Saved raw group episodes to {gen_file}")

    # ---- Phase 2: Annotate (parallel) ----
    ds = EMPATHETIC_DATASET if args.dataset in (EMPATHETIC_DATASET, ESCONV_DATASET) else QMSUM_DATASET

    # Episode-level scoring
    group_evaluator = GroupRubricEvaluator(judge_llm, dataset_name=ds)
    # Turn-level scoring
    individual_evaluator = IndividualRubricEvaluator(judge_llm, dataset_name=ds)

    annotate_group_episodes(all_episode_states, group_evaluator,
                            individual_evaluator, agents, ds, args,
                            _paths["annotated_file"])

    # ---- Phase 3: Format preference pairs ----
    episode_pairs, turn_pairs = build_independent_group_pairs(
        all_episode_states, args, agents, id_key, args.dataset)

    all_pairs = episode_pairs + turn_pairs
    logger.info(f"Created {len(episode_pairs)} episode-level + {len(turn_pairs)} turn-level "
                f"= {len(all_pairs)} total preference pairs")

    # ---- Phase 4: LLM verification ----
    if args.verify_pairs and all_pairs:
        all_pairs = verify_pairs_with_llm(all_pairs, judge_llm)

    # ---- Phase 5: Scale and save ----
    if not all_pairs:
        logger.warning("No preference pairs to save!")
        return

    all_pairs = scale_scores(all_pairs)
    return save_pairs_dataset(all_pairs, _paths["dataset_dir"], id_key=id_key,
                              episode_defaults=True, dataset_name=args.dataset)
