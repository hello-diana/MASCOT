#!/usr/bin/env python3
"""MASCOT evaluation pipeline."""

import os
import json
import argparse
import logging
import random
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from datasets import load_dataset
from collections import OrderedDict
from tqdm import tqdm

from src.const import QMSUM_DATASET, DATASET2HF
from src.arguments import parse_arguments

from src.models.llm_models import VLLMModel, VLLMServerModel, load_vllm_model
from src.prompts.persona_templates import MascotPersonaTemplates
from src.utils.transcript_utils import create_meeting_contexts, format_conversation_history
from src.utils.data_utils import mascot_output_path
from src.generation.mascot_episode import run_mascot_episode_loop, EpisodeConfig
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
) -> OrderedDict:
    """Generate multi-turn QMSum conversations using the shared MASCOT loop."""
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
        ),
        build_directive_prompt=build_directive_prompt,
        build_agent_prompt=build_agent_prompt,
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

    with open(output_file, 'w') as f:
        for state in conversation_states.values():
            # Add metadata
            result = {
                'id': state['id'],
                'start_idx': state['start_idx'],
                'meeting': state['meeting'],
                'turns': state['turns'],
                'agents_spoken': state['agents_spoken'],
                'metadata': {
                    'directive_model': args.directive_model,
                    'agent_model': args.agent_model,
                    'num_turns': num_turns,
                    'directive_temperature': args.directive_temperature,
                    'agent_temperature': args.agent_temperature,
                    'directive_max_tokens': args.directive_max_tokens,
                    'agent_max_tokens': args.agent_max_tokens,
                    'enable_thinking': not args.no_thinking,
                    'mode': args.mode
                }
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

    if args.vllm_server_url:
        # Use external vLLM server via OpenAI-compatible API
        logger.info(f"  Connecting to vLLM server at {args.vllm_server_url}")
        server_model = VLLMServerModel(
            model_name=args.directive_model,
            base_url=args.vllm_server_url,
            enable_thinking=not args.no_thinking,
        )
        directive_generator = server_model
        agent_generator = directive_generator  # same server
    else:
        # Load local vLLM models
        logger.info("  Loading directive model (0.6B)...")
        directive_generator = load_vllm_model(
            model_name_or_path=args.directive_model,
            gpu_memory_utilization=args.directive_gpu_memory,
            tensor_parallel_size=args.directive_tensor_parallel_size,
            enable_lora=(args.directive_lora_path is not None),
            lora_path=args.directive_lora_path
        )

        logger.info("  Loading agent model (8B)...")
        agent_generator = load_vllm_model(
            model_name_or_path=args.agent_model,
            gpu_memory_utilization=args.agent_gpu_memory,
            tensor_parallel_size=args.agent_tensor_parallel_size,
            enable_lora=(args.agent_lora_path is not None and args.mode != "mascot-i"),
            lora_path=None if args.mode == "mascot-i" else args.agent_lora_path
        )

    # Generate conversations
    logger.info("\n[4/5] Generating conversations...")
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
    )

   

    save_results(conversation_states, output_file, args.num_turns, args)

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
