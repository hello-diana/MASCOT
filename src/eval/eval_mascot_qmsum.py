#!/usr/bin/env python3
"""MASCOT evaluation pipeline."""

import os
import json
import argparse
import logging
import random
import time
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from datasets import load_dataset
from collections import OrderedDict
from tqdm import tqdm

from src.const import QMSUM_DATASET, DATASET2HF
from src.arguments import parse_arguments, resolve_server_urls

from src.models.llm_models import VLLMModel, get_shared_clients, resolve_role_identity
from src.prompts.persona_templates import MascotPersonaTemplates
from src.utils.transcript_utils import create_meeting_contexts, format_conversation_history
from src.utils.data_utils import mascot_output_path
from src.utils.run_metadata import (episode_record_metadata, new_counters,
                                    new_run_id, role_identity_metadata,
                                    write_run_config, write_run_summary)
from src.rubrics import PAPER_V1
from src.generation.mascot_episode import run_mascot_episode_loop, EpisodeConfig
from src.user.simulated_users import extract_user_json, generate_user_simulation_prompt_qmsum
from src.utils.run_metadata import record_failure
from config.persona_config import AGENTS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_mascot_episodes(
    meetings: List[Dict],
    directive_generator: VLLMModel,
    agent_generator: VLLMModel,
    num_turns: int = 10,
    directive_temperature: float = 0.9,
    agent_temperature: float = 0.9,
    directive_max_tokens: int = 2048,
    agent_max_tokens: int = 2048,
    enable_thinking: bool = True,
    disable_directive: bool = False,
    num_shot: int = 1,
    strict: bool = False,
    counters: dict = None,
    user_simulator: VLLMModel = None,
    enable_user_simulation: bool = False,
    user_response_probability: float = 0.5,
    user_max_tokens: int = 2048,
    user_temperature: float = 0.9,
) -> OrderedDict:
    """Generate multi-turn QMSum conversations using the shared MASCOT loop."""
    if enable_user_simulation and user_simulator is None:
        raise ValueError("enable_user_simulation=True requires a user_simulator model")
    # Initialize conversation states
    conversation_states = OrderedDict()
    for meeting in meetings:
        key = (meeting['id'], meeting['start_idx'])
        conversation_states[key] = {
            "meeting": meeting,
            "id": meeting['id'],
            "start_idx": meeting['start_idx'],
            "turns": [],
            "agents_spoken": {agent_name: 0 for agent_name in AGENTS.keys()}
        }

    def build_directive_prompt(state):
        history = format_conversation_history(state["turns"], name_key='agent_name')
        return MascotPersonaTemplates.directive_prompt_qmsum(
            agents=AGENTS,
            meeting_context=state["meeting"]["meeting_context"],
            conversation_history=history,
            agents_spoken=state["agents_spoken"],
        )

    def build_agent_prompt(state, directive):
        persona_spec = AGENTS[directive["agent_name"]]
        history = format_conversation_history(state["turns"], name_key='agent_name')
        return MascotPersonaTemplates.speaker_prompt_qmsum(
            persona=persona_spec,
            context=state["meeting"]["meeting_context"],
            meeting_summary=None,
            conversation_history=history,
            num_shot=num_shot,
        )

    # --- Post-turn hook: tag agent turns + optional user simulation -------
    def on_turn_end(conv_states, turn_idx):
        for state in conv_states.values():
            last = state["turns"][-1]
            if "turn_type" not in last:
                last["turn_type"] = "agent"

        if not enable_user_simulation or user_simulator is None:
            return

        keys = [k for k in conv_states
                if random.random() < user_response_probability]
        if not keys:
            return
        user_prompts = [
            generate_user_simulation_prompt_qmsum(
                conv_states[k]["meeting"]["meeting_context"],
                format_conversation_history(conv_states[k]["turns"],
                                            name_key='agent_name'),
            )
            for k in keys
        ]
        user_responses = user_simulator.generate_batch(
            prompts=user_prompts,
            max_tokens=user_max_tokens,
            temperature=user_temperature,
            enable_thinking=False,
            system_prompt=None,
            desc=f"User sim turn {turn_idx + 1} ({len(keys)})",
        )
        for k, raw in zip(keys, user_responses):
            parsed = extract_user_json(raw or "")
            user_text = (parsed.get("response") if parsed else None) or (raw or "").strip()
            if not user_text.strip():
                record_failure(counters, "num_empty_responses", strict,
                               f"empty user response (turn {turn_idx}, conv {k})")
                continue
            conv_states[k]["turns"].append({
                "turn_idx": turn_idx,
                "round_id": turn_idx,
                "turn_type": "user",
                "response": user_text,
            })

    return run_mascot_episode_loop(
        conversation_states=conversation_states,
        directive_generator=directive_generator,
        agent_generator=agent_generator,
        config=EpisodeConfig(
            agents=AGENTS,
            num_turns=num_turns,
            directive_temperature=directive_temperature,
            agent_temperature=agent_temperature,
            directive_max_tokens=directive_max_tokens,
            agent_max_tokens=agent_max_tokens,
            enable_thinking=enable_thinking,
            disable_directive=disable_directive,
            strict=strict,
        ),
        build_directive_prompt=build_directive_prompt,
        build_agent_prompt=build_agent_prompt,
        on_turn_end=on_turn_end,
        counters=counters,
    )


def save_results(
    conversation_states: OrderedDict,
    output_file: Path,
    num_turns: int,
    args: argparse.Namespace
) -> None:
    """Save results to JSONL file (one meeting per line).

    Output format:
    {
        'id': str,
        'start_idx': int,
        'meeting': {...},
        'turns': [
            {
                'turn_idx': int,
                'directive': {
                    'agent_name': str,
                    'action': str,
                    'tone': str
                },
                'agent_name': str,
                'response': str
            },
            ...
        ],
        'agents_spoken': {agent_name: count},
        'metadata': {
            'directive_model': str,
            'agent_model': str,
            'num_turns': int,
            'directive_temperature': float,
            'agent_temperature': float
        }
    }

    Args:
        conversation_states: Generated conversations
        output_file: Path to output JSONL file
        num_turns: Number of turns generated
        args: Command-line arguments for metadata
    """
    logger.info(f"Saving results to {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Base-vs-adapter identity from live client preflight (never fabricated).
    base_models, adapter_extra = role_identity_metadata(
        getattr(args, "_role_identity", {}),
        {"director": args.directive_model, "speaker": args.agent_model,
         "user": args.user_model})

    with open(output_file, 'w') as f:
        for state in conversation_states.values():
            # Add metadata
            result = {
                'id': state['id'],
                'start_idx': state['start_idx'],
                'meeting': state['meeting'],
                'turns': state['turns'],
                'agents_spoken': state['agents_spoken'],
                # Canonical schema block first (evaluation dispatches on it);
                # historical keys are kept alongside for backward compatibility.
                'metadata': episode_record_metadata(
                    system=args.mode,
                    dataset=QMSUM_DATASET,
                    persona_set="default",
                    director_model=base_models["director_model"],
                    speaker_model=base_models["speaker_model"],
                    user_model=(base_models["user_model"]
                                if getattr(args, "enable_user_simulation", False) else None),
                    rubric_version=PAPER_V1,
                    extra={
                        **adapter_extra,
                        'directive_model': args.directive_model,
                        'agent_model': args.agent_model,
                        'num_turns': num_turns,
                        'directive_temperature': args.directive_temperature,
                        'agent_temperature': args.agent_temperature,
                        'directive_max_tokens': args.directive_max_tokens,
                        'agent_max_tokens': args.agent_max_tokens,
                        'enable_thinking': not args.no_thinking,
                        'mode': args.mode,
                    },
                ),
            }
            f.write(json.dumps(result) + '\n')

    logger.info(f"  Saved {len(conversation_states)} conversations")


def get_output_path(args: Namespace) -> Path:
    """Compute the QMSum MASCOT output .jsonl path from parsed args."""
    return mascot_output_path(args, QMSUM_DATASET)


def main(args: Optional[Namespace] = None):
    """Main execution function."""
    # Load environment and parse args
    load_dotenv()
    if args is None:
        args = parse_arguments()
    random.seed(args.seed)

    # Log configuration
    logger.info("=" * 70)
    logger.info("MASCOT Evaluation Pipeline")
    logger.info("=" * 70)
    logger.info(f"Dataset: {args.dataset} ({args.split})")
    logger.info(f"Directive Model: {args.directive_model}")
    logger.info(f"Agent Model: {args.agent_model}")
    logger.info(f"Turns per segment: {args.turns_per_segment}")
    logger.info(f"Conversation turns: {args.num_turns}")
    logger.info("=" * 70)

    # Load dataset
    logger.info("\n[1/5] Loading dataset...")
    dataset = load_dataset(DATASET2HF[args.dataset], split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    logger.info(f"  Loaded {len(dataset)} meetings")

    # Create meeting contexts
    logger.info("\n[2/5] Creating meeting contexts...")
    meetings = create_meeting_contexts(dataset, args.turns_per_segment)
    logger.info(f"  Created {len(meetings)} meeting segments")
    
    # Save results
    logger.info("\n[5/5] Saving results...")
    output_file = get_output_path(args)
    logger.info(f"  Output file: {output_file}")

    # Load models
    logger.info("\n[3/5] Loading models...")

    agent_lora_path = None if args.mode == "mascot-i" else args.agent_lora_path
    role_specs = {
        "director": dict(
            model_name=args.directive_model,
            lora_path=args.directive_lora_path,
            gpu_memory_utilization=args.directive_gpu_memory,
            tensor_parallel_size=args.directive_tensor_parallel_size,
        ),
        "speaker": dict(
            model_name=args.agent_model,
            lora_path=agent_lora_path,
            gpu_memory_utilization=args.agent_gpu_memory,
            tensor_parallel_size=args.agent_tensor_parallel_size,
        ),
    }
    if args.enable_user_simulation:
        role_specs["user"] = dict(
            model_name=args.user_model,
            lora_path=args.user_lora_path,
            gpu_memory_utilization=args.user_gpu_memory,
            tensor_parallel_size=args.user_tensor_parallel_size,
        )
    role_call_trace = [] if args.trace_role_calls else None
    clients = get_shared_clients(role_specs, resolve_server_urls(args),
                                 enable_thinking=not args.no_thinking,
                                 share_model=args.share_model,
                                 trace=role_call_trace)
    directive_generator = clients["director"]
    agent_generator = clients["speaker"]
    user_simulator = clients.get("user")
    args._role_identity = resolve_role_identity(
        clients, {"director": args.directive_model, "speaker": args.agent_model,
                  "user": args.user_model})

    # Default-refuse on existing output; --overwrite replaces it.
    if output_file.exists():
        if not args.overwrite:
            logger.error(f"Output file already exists: {output_file}\n"
                         "Pass --overwrite to replace it.")
            raise SystemExit(2)
        output_file.unlink()
        logger.info(f"--overwrite: removed existing {output_file}")

    # Per-run metadata sidecar: <output>.jsonl.runs/<run_id>/ — repeated runs
    # never overwrite one another's run_config/run_summary.
    model_roles = {"director": args.directive_model, "speaker": args.agent_model}
    if args.enable_user_simulation:
        model_roles["user"] = args.user_model
    run_id = new_run_id(args.mode)
    run_meta_dir = output_file.with_name(output_file.name + ".runs") / run_id
    started_at = time.time()
    served_models = next((c.served_models for c in clients.values()
                          if hasattr(c, "served_models")), None)
    write_run_config(
        run_meta_dir, run_id=run_id, dataset=QMSUM_DATASET,
        system=args.mode, persona_set="default",
        model_roles=model_roles,
        rubric_version=PAPER_V1, args=args, seed=args.seed,
        extra={"output_file": str(output_file),
               "role_identity": args._role_identity,
               "served_models": served_models},
    )

    # Generate conversations
    logger.info("\n[4/5] Generating conversations...")
    counters = new_counters()
    conversation_states = generate_mascot_episodes(
        meetings=meetings,
        directive_generator=directive_generator,
        agent_generator=agent_generator,
        num_turns=args.num_turns,
        directive_temperature=args.directive_temperature,
        agent_temperature=args.agent_temperature,
        directive_max_tokens=args.directive_max_tokens,
        agent_max_tokens=args.agent_max_tokens,
        enable_thinking=not args.no_thinking,
        disable_directive=(args.mode == "mascot-g"),
        num_shot=args.n_shot,
        strict=args.strict,
        counters=counters,
        user_simulator=user_simulator,
        enable_user_simulation=args.enable_user_simulation,
        user_response_probability=args.user_response_probability,
        user_max_tokens=args.user_max_tokens,
        user_temperature=args.user_temperature,
    )

    save_results(conversation_states, output_file, args.num_turns, args)
    extra = {"num_conversations": len(conversation_states),
             "output_file": str(output_file)}
    if args.trace_role_calls and role_call_trace is not None:
        trace_file = run_meta_dir / "role_call_trace.jsonl"
        with open(trace_file, "w") as tf:
            for entry in role_call_trace:
                tf.write(json.dumps(entry) + "\n")
        called_ids = sorted({e["requested_model_id"] for e in role_call_trace})
        extra["role_call_trace"] = str(trace_file)
        extra["called_model_ids"] = called_ids
        logger.info(f"Role-call trace: {len(role_call_trace)} requests to {called_ids} "
                    f"-> {trace_file}")
    write_run_summary(
        run_meta_dir, run_id=run_id, status="completed",
        counters=counters, started_at=started_at, extra=extra,
    )

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("MASCOT Evaluation Complete!")
    logger.info("=" * 70)
    logger.info(f"Processed: {len(conversation_states)} meeting segments")
    logger.info(f"Generated: {args.num_turns} turns each")
    logger.info(f"Output: {output_file}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
