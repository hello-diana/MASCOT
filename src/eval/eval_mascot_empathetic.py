#!/usr/bin/env python3
"""Empathetic dialogue evaluation pipeline with optional user simulation."""

import os
import json
import argparse
import logging
import random
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from collections import OrderedDict
from src.const import EMPATHETIC_DATASET, ESCONV_DATASET
from src.arguments import parse_arguments
from src.models.llm_models import VLLMModel, VLLMServerModel, load_vllm_model
from src.prompts.persona_templates import MascotPersonaTemplates
from src.utils.transcript_utils import format_conversation_history_empathetic
from src.utils.data_utils import mascot_output_path
from src.generation.mascot_episode import run_mascot_episode_loop, EpisodeConfig
from config.persona_config import EMPATHETIC_AGENTS
from src.user.simulated_users import (
    generate_user_simulation_prompt,
    simulate_user_turn,
    build_personality_variants,
    init_empathetic_state,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def generate_empathetic_episodes(
    empathetic_contexts: List[Dict],
    directive_generator: VLLMModel,
    agent_generator: VLLMModel,
    user_simulator: Optional[VLLMModel] = None,
    num_turns: int = 10,
    enable_user_simulation: bool = False,
    user_response_probability: float = 0.5,
    directive_temperature: float = 0.9,
    agent_temperature: float = 0.9,
    user_temperature: float = 0.9,
    directive_max_tokens: int = 4096,
    agent_max_tokens: int = 4096,
    user_max_tokens: int = 2048,
    enable_thinking: bool = True,
    enable_big_five: bool = False,
    user_big_five_profile: Optional[List[str]] = None,
    disable_directive: bool = False,
    num_shot: int = 3,
) -> OrderedDict:
    """Generate multi-turn empathetic conversations using the shared MASCOT loop."""
    if enable_user_simulation and user_simulator is None:
        raise ValueError("enable_user_simulation=True requires user_simulator model")

    # --- Build personality variants list (Big Five profile names, or [None]) ---
    personality_variants = build_personality_variants(
        enable_big_five=enable_big_five, user_big_five_profile=user_big_five_profile,
    )

    # --- Initialize conversation states (with personality expansion) ---
    conversation_states = OrderedDict()

    for context in empathetic_contexts:
        for bf_val in personality_variants:
            key = (context['conv_id'], bf_val)
            conversation_states[key] = init_empathetic_state(
                context, bf_val, EMPATHETIC_AGENTS)

    # --- Prompt builders ---
    def build_directive_prompt(state):
        history = format_conversation_history_empathetic(state["turns"])
        return MascotPersonaTemplates.directive_prompt_empathetic(
            agents=EMPATHETIC_AGENTS,
            situation=state['initial_situation'],
            emotion=state['current_emotion'],
            conversation_history=history,
            agents_spoken=state["agents_spoken"],
        )

    def build_agent_prompt(state, directive):
        persona_spec = EMPATHETIC_AGENTS[directive["agent_name"]]
        history = format_conversation_history_empathetic(state["turns"])
        return MascotPersonaTemplates.speaker_prompt_empathetic(
            persona=persona_spec,
            context=state['initial_situation'],
            conversation_history=history,
            num_shot=num_shot,
        )

    # --- Post-turn hook: tag agent turns + optional user simulation ---
    def on_turn_end(conv_states, turn_idx):
        for state in conv_states.values():
            last = state["turns"][-1]
            if "turn_type" not in last:
                last["turn_type"] = "agent"

        if not enable_user_simulation or user_simulator is None:
            return

        simulate_user_turn(
            conv_states, turn_idx, user_simulator,
            user_max_tokens=user_max_tokens,
            user_temperature=user_temperature,
        )

    # --- Run shared loop ---
    return run_mascot_episode_loop(
        conversation_states=conversation_states,
        directive_generator=directive_generator,
        agent_generator=agent_generator,
        config=EpisodeConfig(
            agents=EMPATHETIC_AGENTS,
            num_turns=num_turns,
            directive_temperature=directive_temperature,
            agent_temperature=agent_temperature,
            directive_max_tokens=directive_max_tokens,
            agent_max_tokens=agent_max_tokens,
            enable_thinking=enable_thinking,
            disable_directive=disable_directive,
        ),
        build_directive_prompt=build_directive_prompt,
        build_agent_prompt=build_agent_prompt,
        on_turn_end=on_turn_end,
    )



def save_empathetic_results(
    conversation_states: OrderedDict,
    output_file: Path,
    args: argparse.Namespace
) -> None:
    """Save empathetic dialogue results to JSONL file.

    Output format:
    {
        'conv_id': str,
        'initial_emotion': str,
        'initial_situation': str,
        'final_emotion': str,
        'emotion_trajectory': List[str],
        'user_participation_enabled': bool,
        'user_participation_count': int,
        'num_turns': int,
        'turns': List[Dict],
        'agents_spoken': Dict[str, int],
        'metadata': Dict
    }

    Args:
        conversation_states: Generated conversations
        output_file: Path to output JSONL file
        args: Command-line arguments for metadata
    """
    logger.info(f"Saving results to {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Map internal persona keys to friendly names for output
    _AGENT_NAME_MAP = {
        "the_anchor": "alice",
        "the_beacon": "bob",
        "the_catalyst": "cathy",
    }

    def _remap_agent_name(name):
        return _AGENT_NAME_MAP.get(name, name)

    with open(output_file, 'w') as f:
        for state in conversation_states.values():
            # Remap agent names in turns
            remapped_turns = []
            for turn in state['turns']:
                t = dict(turn)
                if 'agent_name' in t:
                    t['agent_name'] = _remap_agent_name(t['agent_name'])
                if 'directive' in t and isinstance(t['directive'], dict):
                    t['directive'] = dict(t['directive'])
                    if 'agent_name' in t['directive']:
                        t['directive']['agent_name'] = _remap_agent_name(t['directive']['agent_name'])
                remapped_turns.append(t)

            # Remap agent names in agents_spoken
            remapped_agents_spoken = {
                _remap_agent_name(k): v for k, v in state['agents_spoken'].items()
            }

            # Build result dict
            result = {
                'conv_id': state['conv_id'],
                'initial_emotion': state['initial_emotion'],
                'initial_situation': state['initial_situation'],
                'final_emotion': state['current_emotion'],
                'emotion_trajectory': state['emotion_trajectory'],
                'user_participation_enabled': args.enable_user_simulation,
                'user_participation_count': state['user_participation_count'],
                'user_big_five': state.get('user_big_five'),
                'num_turns': len([t for t in state['turns'] if t.get('turn_type') == 'agent']),
                'turns': remapped_turns,
                'agents_spoken': remapped_agents_spoken,
                'metadata': {
                    'directive_model': args.directive_model,
                    'agent_model': args.agent_model,
                    'user_model': args.user_model if args.enable_user_simulation else None,
                    'num_turns': args.num_turns,
                    'user_response_probability': args.user_response_probability,
                    'directive_temperature': args.directive_temperature,
                    'agent_temperature': args.agent_temperature,
                    'user_temperature': args.user_temperature,
                    'directive_max_tokens': args.directive_max_tokens,
                    'agent_max_tokens': args.agent_max_tokens,
                    'user_max_tokens': args.user_max_tokens,
                    'enable_thinking': not args.no_thinking,
                    'enable_big_five': args.enable_big_five,
                    'user_big_five_profile': args.user_big_five_profile,
                    'mode': args.mode
                }
            }
            f.write(json.dumps(result) + '\n')

    logger.info(f"  Saved {len(conversation_states)} conversations")


def get_output_path(args: Namespace) -> Path:
    """Compute the Empathetic/ESConv MASCOT output .jsonl path from parsed args."""
    dataset = ESCONV_DATASET if getattr(args, "dataset", None) == ESCONV_DATASET else EMPATHETIC_DATASET
    return mascot_output_path(args, dataset)


def main(args: Optional[Namespace] = None):
    """Main execution function."""
    # Load environment and parse args
    load_dotenv()
    if args is None:
        args = parse_arguments()
    # ESConv path: use the ESConv adapter (returns Empathetic-flat utterances)
    # Activated by passing --dataset ESConv.
    

    # Set random seed
    random.seed(args.seed)

    # Log configuration
    logger.info("=" * 70)
    logger.info("Empathetic Dialogue Evaluation Pipeline")
    logger.info("=" * 70)
    logger.info(f"Dataset: {args.dataset} ({args.split})")
    logger.info(f"Mode: {'User Simulation Enabled' if args.enable_user_simulation else 'Agent-Only'}")
    if args.enable_user_simulation:
        logger.info(f"User Response Probability: {args.user_response_probability}")
    if args.enable_big_five:
        if args.user_big_five_profile:
            logger.info(f"Big Five: Specific profiles ({', '.join(args.user_big_five_profile)})")
        else:
            logger.info(f"Big Five: All 10 profiles (High/Low x 5 traits)")
    logger.info(f"Directive Model: {args.directive_model}")
    logger.info(f"Agent Model: {args.agent_model}")
    if args.enable_user_simulation:
        logger.info(f"User Model: {args.user_model}")
    logger.info(f"Conversation turns: {args.num_turns}")
    logger.info("=" * 70)

    # [1/4] Load empathetic contexts (one dict per conversation)
    logger.info("\n[1/4] Loading empathetic contexts...")
    if args.dataset == ESCONV_DATASET:
        from src.utils.data_utils import load_esconv_situations
        empathetic_contexts = load_esconv_situations(
            split=args.split, max_samples=args.max_samples,
        )
    elif args.dataset == EMPATHETIC_DATASET:
        from src.utils.data_utils import load_empathetic_situations
        empathetic_contexts = load_empathetic_situations(
            split=args.split, max_samples=args.max_samples,
        )
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    logger.info(f"  Loaded {len(empathetic_contexts)} contexts")

    # [2/4] Load models
    logger.info("\n[2/4] Loading models...")


    # TODO
    if args.vllm_server_url:
        assert args.share_model, "share_model must be True when using vllm_server_url"
        # Use external vLLM server via OpenAI-compatible API
        logger.info(f"  Connecting to vLLM server at {args.vllm_server_url}")
        server_model = VLLMServerModel(
            model_name=args.directive_model,
            base_url=args.vllm_server_url,
            enable_thinking=not args.no_thinking,
        )
        directive_generator = server_model
        agent_generator = directive_generator  

        user_simulator = None
        if args.enable_user_simulation:
            logger.info("  User simulator shares vLLM server")
            user_simulator = directive_generator
    else:
        # Load local vLLM models
        agent_lora_path = None if args.mode == "mascot-i" else args.agent_lora_path

        if args.share_model:
            logger.info("  Loading shared model (directive + agent)...")
            agent_generator = load_vllm_model(
                model_name_or_path=args.agent_model,
                gpu_memory_utilization=args.agent_gpu_memory,
                tensor_parallel_size=args.agent_tensor_parallel_size,
                enable_lora=(args.agent_lora_path is not None and args.mode != "mascot-i"),
                lora_path=agent_lora_path
            )
            directive_generator = agent_generator
        else:
            logger.info("  Loading directive model...")
            directive_generator = load_vllm_model(
                model_name_or_path=args.directive_model,
                gpu_memory_utilization=args.directive_gpu_memory,
                tensor_parallel_size=args.directive_tensor_parallel_size,
                enable_lora=(args.directive_lora_path is not None),
                lora_path=args.directive_lora_path
            )

            if (args.agent_model == args.directive_model
                    and agent_lora_path == args.directive_lora_path):
                logger.info("  Agent model shares directive model instance")
                agent_generator = directive_generator
            else:
                logger.info("  Loading agent model...")
                agent_generator = load_vllm_model(
                    model_name_or_path=args.agent_model,
                    gpu_memory_utilization=args.agent_gpu_memory,
                    tensor_parallel_size=args.agent_tensor_parallel_size,
                    enable_lora=(args.agent_lora_path is not None and args.mode != "mascot-i"),
                    lora_path=agent_lora_path
                )

        user_simulator = None
        if args.enable_user_simulation:
            logger.info("  Loading user simulator model...")
            user_simulator = load_vllm_model(
                model_name_or_path=args.user_model,
                gpu_memory_utilization=args.user_gpu_memory,
                tensor_parallel_size=args.user_tensor_parallel_size,
                enable_lora=(args.user_lora_path is not None),
                lora_path=args.user_lora_path
            )

    # [3/4] Generate conversations
    logger.info("\n[3/4] Generating empathetic conversations...")
    conversation_states = generate_empathetic_episodes(
        empathetic_contexts=empathetic_contexts,
        directive_generator=directive_generator,
        agent_generator=agent_generator,
        user_simulator=user_simulator,
        num_turns=args.num_turns,
        enable_user_simulation=args.enable_user_simulation,
        user_response_probability=args.user_response_probability,
        directive_temperature=args.directive_temperature,
        agent_temperature=args.agent_temperature,
        user_temperature=args.user_temperature,
        directive_max_tokens=args.directive_max_tokens,
        agent_max_tokens=args.agent_max_tokens,
        user_max_tokens=args.user_max_tokens,
        enable_thinking=not args.no_thinking,
        enable_big_five=args.enable_big_five,
        user_big_five_profile=args.user_big_five_profile,
        disable_directive=(args.mode == "mascot-g"),
        num_shot=args.n_shot,
    )

    # [4/4] Save results
    logger.info("\n[4/4] Saving results...")
    output_file = get_output_path(args)

    save_empathetic_results(conversation_states, output_file, args)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Empathetic Dialogue Evaluation Complete!")
    logger.info("=" * 70)
    logger.info(f"Processed: {len(conversation_states)} conversations")
    logger.info(f"Generated: {args.num_turns} turns each")
    logger.info(f"Output: {output_file}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
