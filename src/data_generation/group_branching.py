"""Branching group-episode generation pipeline (turn-centric batching).

Moved verbatim from src/dataset/generate_and_annotate.py (Phase-2 Part A).
The per-dataset episode hooks are shared with the independent pipeline
(src/data_generation/group_independent.py).
"""

import logging
import os
import random
from collections import OrderedDict
from typing import List

from datasets import load_dataset

from config.persona_config import AGENTS, EMPATHETIC_AGENTS
from src.const import (DATASET2HF, EMPATHETIC_DATASET, ESCONV_DATASET,
                       QMSUM_DATASET)
from src.data_generation.annotation import (annotate_branching_states,
                                            verify_pairs_with_llm)
from src.data_generation.common import artifact_paths, scale_scores
from src.data_generation.group_independent import (_build_agent_emp,
                                                   _build_agent_qmsum,
                                                   _build_directive_emp,
                                                   _build_directive_qmsum,
                                                   _on_turn_end_emp,
                                                   _on_turn_end_qmsum)
from src.data_generation.preference_pairs import build_branching_pairs
from src.data_generation.serialization import (read_jsonl, save_pairs_dataset,
                                               write_jsonl)
from src.eval.rubric_evaluator import GroupRubricEvaluator
from src.generation.mascot_episode import parse_directive
from src.models.llm_models import VLLMServerModel
from src.utils.data_utils import (load_empathetic_situations,
                                  load_esconv_situations)
from src.utils.transcript_utils import create_meeting_contexts

logger = logging.getLogger(__name__)


def group_conversation_pipeline_with_branching(args, llm: VLLMServerModel, judge_llm: VLLMServerModel):
    """Generate branching episodes with turn-centric batching.

    Architecture:
    - Pre-assign branch_turn_idx for ALL meetings
    - Maintain conversation_states: OrderedDict[(meeting_idx, candidate_idx)] → state
    - Each turn: single generate_batch for directives, single generate_batch for agents
    - At branch turn: fork states into num_candidates copies with diverse directives
    - Single batched evaluation across ALL episodes at the end

    Supports both QMSum and Empathetic datasets.
    """
    agents = AGENTS if args.dataset == QMSUM_DATASET else EMPATHETIC_AGENTS  # ESConv reuses empathetic personas
    id_key = "meeting_id" if args.dataset == QMSUM_DATASET else "conv_id"
    num_candidates = args.num_candidates
    num_turns = args.num_turns

    gen_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(gen_dir, exist_ok=True)
    _paths = artifact_paths(args)
    gen_file = _paths["raw_file"]

    if os.path.exists(gen_file):
        logger.info(f"Found existing branching data at {gen_file}, skipping generation")
        all_query_data: List[dict] = read_jsonl(gen_file)
    else:
        # ---- Load dataset ----
        if args.dataset == QMSUM_DATASET:
            dataset = load_dataset(DATASET2HF[QMSUM_DATASET], split="train")
            if args.max_samples:
                dataset = dataset.select(range(min(args.max_samples, len(dataset))))
            items = create_meeting_contexts(dataset, args.turns_per_round)
        else:
            if args.dataset == ESCONV_DATASET:
                items = load_esconv_situations(split="train", max_samples=args.max_samples)
            else:
                items = load_empathetic_situations(split="train", max_samples=args.max_samples)

        # Pre-assign branch positions for ALL items
        branch_turn_indices = {m_idx: random.randint(0, num_turns - 1)
                               for m_idx in range(len(items))}

        # One state per item, keyed by (item_idx, candidate_idx)
        conversation_states = OrderedDict()
        for m_idx, item in enumerate(items):
            if args.dataset == QMSUM_DATASET:
                state = {
                    "meeting": item, "id": item["id"],
                    "start_idx": item["start_idx"],
                    "turns": [], "agents_spoken": {n: 0 for n in agents},
                }
            else:
                state = {
                    "conv_id": item["conv_id"],
                    "initial_emotion": item["emotion"],
                    "initial_situation": item["situation"].replace("__comma__", ", "),
                    "current_emotion": item["emotion"],
                    "turns": [], "agents_spoken": {n: 0 for n in agents},
                }
            conversation_states[(m_idx, 0)] = state

        logger.info(f"Starting turn-centric branching: {len(items)} items, "
                     f"{num_turns} turns, {num_candidates} candidates")

        # ---- Set dataset-specific function references ----
        _build_directive = (lambda s: _build_directive_qmsum(s, agents)) if args.dataset == QMSUM_DATASET \
                           else (lambda s: _build_directive_emp(s, agents))
        _build_agent = (lambda s, d: _build_agent_qmsum(s, d, agents)) if args.dataset == QMSUM_DATASET \
                       else (lambda s, d: _build_agent_emp(s, d, agents))
        _on_turn_end = (lambda cs, ti: _on_turn_end_qmsum(cs, ti, llm, args.user_response_probability)) \
                       if args.dataset == QMSUM_DATASET else \
                       (lambda cs, ti: _on_turn_end_emp(cs, ti, llm, args.user_response_probability))

        branch_directive_prompts = {}  # m_idx -> directive prompt at branch turn
        completed_states = OrderedDict()  # (m_idx, c_idx) → state, for meetings done branching

        # ---- Turn-centric loop ----
        for turn_idx in range(num_turns):
            keys = list(conversation_states.keys())

            # D1. Partition: which meetings branch at this turn?
            branching_keys = [k for k in keys
                              if branch_turn_indices[k[0]] == turn_idx and k[1] == 0]
            branching_set = set(branching_keys)
            normal_keys = [k for k in keys if k not in branching_set]

            logger.info(f"Turn {turn_idx}: {len(keys)} states "
                         f"({len(branching_keys)} branching, {len(normal_keys)} normal)")

            # D2. Build directive prompts
            directive_prompts = []
            prompt_source = []  # (key, candidate_idx_or_None)

            for key in branching_keys:
                prompt = _build_directive(conversation_states[key])
                branch_directive_prompts[key[0]] = prompt
                for c in range(num_candidates):
                    directive_prompts.append(prompt)
                    prompt_source.append((key, c))

            for key in normal_keys:
                directive_prompts.append(
                    _build_directive(conversation_states[key]))
                prompt_source.append((key, None))

            # SINGLE batch call for all directives
            raw_results = llm.generate_batch(
                prompts=directive_prompts, max_tokens=4096, temperature=1.0,
                desc=f"Directives turn {turn_idx} ({len(directive_prompts)})",
            ) if directive_prompts else []

            directive_outputs = [(t or "").strip() for t in raw_results]

            # D3. Parse directives — NO random fallback
            parsed_directives = [parse_directive(output, agents)
                                 for output in directive_outputs]

            # Retry failed non-branching directives up to directive_retries times
            branching_count = len(branching_keys) * num_candidates
            for retry_num in range(args.directive_retries):
                retry_indices = []
                retry_prompts = []
                for i in range(branching_count, len(parsed_directives)):
                    if parsed_directives[i] is None:
                        retry_indices.append(i)
                        retry_prompts.append(directive_prompts[i])

                if not retry_prompts:
                    break

                retry_results = llm.generate_batch(
                    prompts=retry_prompts, max_tokens=4096, temperature=1.0,
                    desc=f"Directive retry {retry_num+1}/{args.directive_retries} ({len(retry_prompts)})",
                )
                for j, idx in enumerate(retry_indices):
                    parsed_directives[idx] = parse_directive(
                        (retry_results[j] or "").strip(), agents)

            # D4. Drop meetings with unrecoverable failures
            dropped_meetings = set()

            # Non-branching: drop if directive still None after retry
            for i in range(branching_count, len(parsed_directives)):
                if parsed_directives[i] is None:
                    key, _ = prompt_source[i]
                    dropped_meetings.add(key[0])

            # Branching: need >=2 valid directives per meeting
            branching_dirs = {}  # key → list of valid directives
            offset = 0
            for key in branching_keys:
                m_idx = key[0]
                dirs_for_meeting = []
                for c in range(num_candidates):
                    d = parsed_directives[offset + c]
                    if d is not None:
                        dirs_for_meeting.append(d)
                offset += num_candidates
                if len(dirs_for_meeting) < 2:
                    logger.warning(f"Meeting {m_idx}: only {len(dirs_for_meeting)} "
                                   f"valid directives at branch turn, dropping")
                    dropped_meetings.add(m_idx)
                else:
                    branching_dirs[key] = dirs_for_meeting

            # Remove dropped meetings from conversation_states
            if dropped_meetings:
                for key in list(conversation_states.keys()):
                    if key[0] in dropped_meetings:
                        del conversation_states[key]
                branching_keys = [k for k in branching_keys
                                  if k[0] not in dropped_meetings]
                normal_keys = [k for k in normal_keys
                               if k[0] not in dropped_meetings]
                logger.info(f"  Dropped {len(dropped_meetings)} meetings "
                             f"due to parse failures")

            # Build normal directive map from surviving states
            normal_directive_map = {}
            for i, (key, candidate) in enumerate(prompt_source):
                if candidate is None and key[0] not in dropped_meetings:
                    normal_directive_map[key] = parsed_directives[i]

            # D5. Fork branching states into candidate copies
            for key in branching_keys:
                m_idx = key[0]
                prefix_state = conversation_states[key]
                actual_num = len(branching_dirs[key])
                for c in range(1, actual_num):
                    fork = {k: v for k, v in prefix_state.items()
                            if k not in ("turns", "agents_spoken")}
                    fork["turns"] = [dict(t) for t in prefix_state["turns"]]
                    fork["agents_spoken"] = dict(prefix_state["agents_spoken"])
                    conversation_states[(m_idx, c)] = fork

            # D6. Build agent prompts for ALL active states
            agent_prompts = []
            agent_keys = []
            agent_directive_list = []

            for key in branching_keys:
                m_idx = key[0]
                dirs = branching_dirs[key]
                for c, directive in enumerate(dirs):
                    state_key = (m_idx, c)
                    agent_prompts.append(
                        _build_agent(
                            conversation_states[state_key], directive))
                    agent_keys.append(state_key)
                    agent_directive_list.append(directive)

            for key in normal_keys:
                directive = normal_directive_map[key]
                agent_prompts.append(
                    _build_agent(conversation_states[key], directive))
                agent_keys.append(key)
                agent_directive_list.append(directive)

            # SINGLE batch call for all agent responses
            agent_results = llm.generate_batch(
                prompts=agent_prompts, max_tokens=4096, temperature=0.9,
                desc=f"Agents turn {turn_idx} ({len(agent_prompts)})",
            ) if agent_prompts else []

            # D7. Update states with agent responses
            for i in range(len(agent_keys)):
                state = conversation_states[agent_keys[i]]
                directive = agent_directive_list[i]
                state["agents_spoken"][directive["agent_name"]] += 1
                state["turns"].append({
                    "turn_idx": turn_idx, "round_id": turn_idx, "directive": directive,
                    "agent_name": directive["agent_name"],
                    "response": (agent_results[i] or "").strip(), "turn_type": "agent",
                })

            # D8. User simulation (already batches internally)
            _on_turn_end(conversation_states, turn_idx)

            # D9. Early-stop: move completed branching meetings out of active loop
            if args.stop_after_branch:
                for key in branching_keys:
                    m_idx = key[0]
                    if m_idx in dropped_meetings:
                        continue
                    actual_num = len(branching_dirs.get(key, []))
                    for c in range(actual_num):
                        state_key = (m_idx, c)
                        if state_key in conversation_states:
                            completed_states[state_key] = conversation_states.pop(state_key)

        # Merge any remaining states (meetings that never branched in the loop)
        for key, state in conversation_states.items():
            completed_states[key] = state

        # ---- Batched evaluation ----
        group_evaluator = GroupRubricEvaluator(judge_llm, dataset_name=QMSUM_DATASET if args.dataset == QMSUM_DATASET else EMPATHETIC_DATASET)  # ESConv reuses empathetic rubric
        annotate_branching_states(completed_states, group_evaluator, judge_llm,
                                  agents, args)

        # ---- Build query records ----
        meeting_episodes = {}
        for (m_idx, c_idx), state in completed_states.items():
            meeting_episodes.setdefault(m_idx, []).append((c_idx, state))

        all_query_data = []
        for m_idx in sorted(meeting_episodes):
            item = items[m_idx]
            candidates = sorted(meeting_episodes[m_idx], key=lambda x: x[0])
            episodes = [state for _, state in candidates]
            rec = {
                "branch_turn_idx": branch_turn_indices[m_idx],
                "num_candidates": len(episodes),
                "episodes": episodes,
            }
            if args.dataset == QMSUM_DATASET:
                rec["query_key"] = f"{item['id']}_{item['start_idx']}"
                rec["meeting_id"] = item["id"]
                rec["start_idx"] = item["start_idx"]
            else:
                rec["query_key"] = item["conv_id"]
                rec["conv_id"] = item["conv_id"]
            rec["branch_directive_prompt"] = branch_directive_prompts.get(m_idx, "")
            all_query_data.append(rec)

        # Save raw branching data
        write_jsonl(gen_file, all_query_data)
        logger.info(f"Saved raw branching data to {gen_file}")

    # ---- Phase 2: Create preference pairs ----
    pairs = build_branching_pairs(all_query_data, args, id_key)

    # ---- Phase 3: LLM verification ----
    if args.verify_pairs and pairs:
        pairs = verify_pairs_with_llm(pairs, judge_llm)

    # ---- Phase 4: Scale and save ----
    if not pairs:
        logger.warning("No branching preference pairs to save!")
        return

    pairs = scale_scores(pairs)
    save_path = save_pairs_dataset(pairs, _paths["dataset_dir"],
                                   both_id_keys=True, episode_defaults=True,
                                   dataset_name=args.dataset)

    # Print K distribution stats
    branch_pos_counts = {}
    for rec in all_query_data:
        branch_pos = rec["branch_turn_idx"]
        branch_pos_counts[branch_pos] = branch_pos_counts.get(branch_pos, 0) + 1
    logger.info(f"Branch position distribution: {dict(sorted(branch_pos_counts.items()))}")

    return save_path
