#!/usr/bin/env python3
"""Batch evaluation script for MASCOT conversations.

This script evaluates pre-generated MASCOT conversations using rubric-based
evaluators. It performs both individual turn-by-turn evaluation and group-level
conversation evaluation with configurable turn ranges.
"""

import sys
import os
import traceback

from tqdm import tqdm
import json
import argparse
import logging
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

from src.const import EMPATHETIC_DATASET, ESCONV_DATASET, QMSUM_DATASET, DATASETS, DEFAULT_JUDGE_MODEL
from src.models.llm_models import GPT, Claude, Gemini, VLLMServerModel, parse_json_response
from config.persona_config import AGENTS, EMPATHETIC_AGENTS
from src.eval.rubric_evaluator import IndividualRubricEvaluator, GroupRubricEvaluator
from src.utils.data_utils import get_file_timestamps, judge_metrics_path
from src.utils.eval_utils import print_colored
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _preparse_structured(texts, judge_model):
    """Reproduce the former server-side structured-output pre-parse.

    VLLMServerModel used to return BatchResult with parsed_json =
    parse_json_response(text) (response_format was always passed here); the
    evaluators consumed that dict first. GPT/Claude/Gemini judges returned
    plain strings and never had the pre-parse — they are passed through.
    """
    if not isinstance(judge_model, VLLMServerModel):
        return list(texts)
    out = []
    for t in texts:
        parsed = parse_json_response(t)
        out.append(parsed if isinstance(parsed, dict) else t)
    return out

logging.getLogger("httpx").setLevel(logging.WARNING)


def load_mascot_conversations(input_file: Path) -> List[Dict]:
    """Load MASCOT conversations from JSONL file.

    Args:
        input_file: Path to MASCOT JSONL file

    Returns:
        List of conversation dicts with structure:
        {
            'id': str,
            'start_idx': int,
            'meeting': {...},
            'turns': List[Dict],
            'agents_spoken': Dict[str, int],
            'metadata': {...}
        }
    """
    conversations = []
    with open(input_file, 'r') as f:
        for line in f:
            conversations.append(json.loads(line))

    logger.info(f"Loaded {len(conversations)} conversations")
    return conversations


def clean_response(response: str) -> str:
    """Remove <think></think> tags from MASCOT responses.

    Args:
        response: Raw response text (may contain thinking tags)

    Returns:
        Cleaned response without thinking tags
    """
    if "</think>" in response:
        return response.split("</think>", 1)[-1].strip()
    return response.strip()


def extract_turn_range(
    turns: List[Dict],
    start_idx: int,
    end_idx: Optional[int]
) -> List[Dict]:
    """Extract a specific range of turns from conversation.

    Args:
        turns: Full list of conversation turns
        start_idx: Starting turn index (inclusive, 0-based)
        end_idx: Ending turn index (exclusive, 0-based, None = all turns)

    Returns:
        List of turns in the specified range

    Example:
        extract_turn_range(turns, 0, 5) -> turns[0:5] (turns 0-4)
        extract_turn_range(turns, 2, 8) -> turns[2:8] (turns 2-7)
    """
    if end_idx is None or end_idx > len(turns):
        return turns
        
    else:
        count = 0
        for i, turn in enumerate(turns):
            if turn.get('turn_type', None) != 'user':
                count += 1
            if count == end_idx:
                return turns[:i+1]
            
                
        return turns



def build_conversation_history(
    meeting_context: str,
    turns: List[Dict],
    up_to_turn: int,
    dataset_name: str
) -> str:
    """Build conversation history up to a specific turn.

    This is used for individual evaluations to provide context for each turn.

    Args:
        meeting_context: Original meeting transcript
        turns: List of conversation turns
        up_to_turn: Build history up to (but not including) this turn index

    Returns:
        Formatted string with meeting context and conversation history
    """
    if up_to_turn == 0:
        return meeting_context

    # Build history from turns 0 to up_to_turn-1
    history_turns = turns[:up_to_turn]
    conversation_text = ""
    
    if dataset_name == QMSUM_DATASET:
        conversation_text = "\n".join([
            f"{turn['agent_name']}: {clean_response(turn['response'])}"
            for turn in history_turns
        ])
    elif dataset_name == EMPATHETIC_DATASET:
        for turn in history_turns:
            if turn['turn_type'] == 'agent':
                conversation_text += f"[{turn['agent_name']} agent]: {clean_response(turn['response'])}\n\n"
            elif turn['turn_type'] == 'user':
                conversation_text += f"[User]: {clean_response(turn['response'])}\n\n"

    return f"{meeting_context}\n\n## Agent Responses in Previous Turns:\n{conversation_text}"


def prepare_individual_evaluation_tasks(
    args,
    conversations: List[Dict],
    turn_start: int,
    turn_end: Optional[int],
    agents: List[Dict],
    dataset_name: str
) -> Tuple[List[List[Dict]], Dict]:
    """Prepare individual evaluation tasks for batch processing.

    For each conversation, create a batch of evaluation tasks (one per turn)
    that includes the progressive conversation history.

    Args:
        conversations: List of MASCOT conversation dicts
        turn_start: Start turn index (inclusive)
        turn_end: End turn index (exclusive, None = all turns)
        agents: Dictionary of agent configurations
        dataset_name: Dataset type ('qmsum' or 'empathetic')

    Returns:
        Tuple of:
        - List of conversation batches (outer list = conversations, inner list = turns)
        - Metadata dict for reconstruction (maps back to original structure)
    """
    conversation_batches = []  # List[List[Dict]] - one batch per conversation
    metadata = {
        'mapping': []  # List of (conv_idx, turn_idx) tuples
    }

    for conv_idx, conv in enumerate(conversations):
        # Extract context based on dataset type
        meeting_context = conv['meeting']['meeting_context'] if dataset_name == QMSUM_DATASET else conv['initial_situation']
        turns = extract_turn_range(conv['turns'], turn_start, turn_end)

        # Filter to agent turns only (for empathetic, removes user turns)
        agent_turns = turns if dataset_name == QMSUM_DATASET else [turn for turn in turns if turn.get('turn_type') == 'agent']
        if not agent_turns:
            logger.warning(f"No agent turns found in conversation {conv.get('id', conv_idx)}")
            continue
        turns = agent_turns

        # Build batch for THIS conversation only
        conv_batch = []
        for local_turn_idx, turn in enumerate(turns):
            # Global turn index in original conversation
            global_turn_idx = turn_start + local_turn_idx

            # Build context up to this turn (includes meeting + prior turns)
            conversation_context = build_conversation_history(
                meeting_context, conv['turns'], global_turn_idx, dataset_name
            )

            # Get agent persona
            agent_name = turn['agent_name']
            
            
                
            if "mascot" in os.path.basename(args.input_file).lower():
                # Empathetic eval output remaps the_anchor/the_beacon/the_catalyst
                # to alice/bob/cathy for friendlier output. Resolve back here.
                _EMP_ALIAS = {"alice": "the_anchor", "bob": "the_beacon", "cathy": "the_catalyst"}
                _key = _EMP_ALIAS.get(agent_name, agent_name)
                persona_spec = agents.get(_key) or agents[agent_name]
                
                agent_persona_full = f"""Agent: {persona_spec.name}
Role: {persona_spec.meeting_role}
Description: {persona_spec.description}

Key Traits: {', '.join(persona_spec.traits)}"""

            else: 
                agent_persona_full = f"""Agent: {agent_name}
"""

            # Clean response
            clean_resp = clean_response(turn['response'])
            
            if dataset_name == QMSUM_DATASET:

                # Add to this conversation's batch
                conv_batch.append({
                    'id': conv['id'],
                    'start_idx': conv['start_idx'],
                    'agent_persona': agent_persona_full,
                    'conversation_context': conversation_context,
                    'agent_response': clean_resp,
                    'agent_name': agent_name,
                    'response_idx': global_turn_idx
                })
            elif dataset_name == EMPATHETIC_DATASET:
                conv_batch.append({
                    'id': conv['conv_id'],
                    'agent_persona': agent_persona_full,
                    'conversation_context': conversation_context,
                    'agent_response': clean_resp,
                    'agent_name': agent_name,
                    'response_idx': global_turn_idx
                })

            metadata['mapping'].append((conv_idx, global_turn_idx))

        conversation_batches.append(conv_batch)

    total_tasks = sum(len(batch) for batch in conversation_batches)
    logger.info(f"Prepared {total_tasks} individual evaluation tasks across {len(conversation_batches)} conversations")
    return conversation_batches, metadata



def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch evaluation for MASCOT conversations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument(
        "--input-file", type=str, required=True,
        help="Path to MASCOT JSONL file (e.g., outputs/mascot_eval/.../mascot_*.jsonl)"
    )
    io_group.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: same as input file directory)"
    )

    # Turn Range
    turn_group = parser.add_argument_group('Turn Range')
    turn_group.add_argument(
        "--eval-turns-start", type=int, default=0,
        help="Start turn index for evaluation (inclusive, 0-based)"
    )
    turn_group.add_argument(
        "--eval-turns-end", type=int, default=None,
        help="End turn index for evaluation (exclusive, None = all turns)"
    )
    turn_group.add_argument(
        "--batch-size", type=int, default=2,
        help="Batch size for group evaluation ONLY"
    )

    # Evaluation Flags
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument(
        "--judge", type=str, default=DEFAULT_JUDGE_MODEL,
        help="Judge model name (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022', 'gemini-1.5-flash')"
    )
    eval_group.add_argument(
        "--dataset", type=str, required=True,
        choices=list(DATASETS),
        help=f"Dataset name: one of {list(DATASETS)} (ESConv reuses empathetic rubric)"
    )
    eval_group.add_argument(
        "--enable-big-five", action="store_true",
        help="Enable Big Five (OCEAN) personality profiles for evaluation"
    )
    eval_group.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of conversations to evaluate (default: all)"
    )
    eval_group.add_argument(
        "--vllm-judge-url", type=str, default=None,
        help="URL of vLLM server for judge model (e.g., http://localhost:8000/v1). "
             "When set, uses VLLMServerModel with --judge as model name."
    )

    # Judge prompt perturbation
    perturb_group = parser.add_argument_group('Judge Prompt Perturbation')
    perturb_group.add_argument(
        "--criteria-order-individual", type=parse_criteria_order, default=None,
        help="Comma-separated permutation of 0-4 reordering the individual rubric "
             "criteria in the judge prompt (default: original order)"
    )
    perturb_group.add_argument(
        "--criteria-order-group", type=parse_criteria_order, default=None,
        help="Comma-separated permutation of 0-3 reordering the group rubric "
             "criteria in the judge prompt (default: original order)"
    )
    perturb_group.add_argument(
        "--output-suffix", type=str, default="",
        help="Suffix appended to the output filename before '.json' "
             "(e.g., '_perturb-reversed') so perturbed runs don't overwrite originals"
    )
    return parser.parse_args()


def parse_criteria_order(s: Optional[str]) -> Optional[list]:
    """Parse a comma-separated criteria permutation like '4,3,2,1,0'."""
    if s is None or s == "":
        return None
    return [int(x) for x in s.split(",")]


def main(args: Optional[Namespace] = None):
    """Main execution function."""
    # 1. Parse arguments and setup
    if args is None:
        args = parse_arguments()
    load_dotenv()
    # ESConv reuses the empathetic rubric internally; output filenames retain
    # ESConv identity via the input-file path (e.g., outputs/mascot_eval/ESConv/test/...).
    if args.dataset == ESCONV_DATASET:
        args.dataset = EMPATHETIC_DATASET
    if "big5" in args.input_file:
        assert args.enable_big_five, "Big Five is not enabled"

    # Determine personality mode
    enable_personality = args.enable_big_five
    personality_field = 'user_big_five'

    # Validation
    input_file = Path(args.input_file)
    if not input_file.exists():
        
        logger.error(f"Input file not found: {input_file}")
        return
    
    timestamps = get_file_timestamps(input_file)
    logger.info(f"Loading OUTPUT FILE from: {input_file.name}")
   

    logger.info("=" * 70)
    logger.info("MASCOT Batch Evaluation Pipeline")
    logger.info("=" * 70)
    logger.info(f"Input file: {input_file}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Evaluation: Both individual and group")
    if args.enable_big_five:
        logger.info("Big Five: Enabled")
    logger.info("=" * 70)

    # Determine output path early to check for partial results
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_file.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    turn_end = args.eval_turns_end

    # Include dataset name in output filename for clarity
    output_suffix = getattr(args, 'output_suffix', '') or ''
    output_file = judge_metrics_path(input_file, args.judge, output_dir=output_dir,
                                     suffix=output_suffix)

    # Judge prompt perturbation settings (absent when main() is called with an
    # externally built Namespace, e.g. from run_eval_mascot.py)
    criteria_order_individual = getattr(args, 'criteria_order_individual', None)
    criteria_order_group = getattr(args, 'criteria_order_group', None)

    # Initialize conversations structure
    conversations = {
        "group": [],
        "individual": [],
        "metadata": {
            "input_file": str(input_file),
            "judge": args.judge,
            "criteria_order_individual": criteria_order_individual,
            "criteria_order_group": criteria_order_group,
            "command": " ".join(sys.argv),
        }
    }

    # Try to load existing results
    if output_file.exists():
        timestamps = get_file_timestamps(output_file)
        logger.info(f"Loading OUTPUT FILE from: {output_file}")
        logger.info(f"  Created: {timestamps['created_str']}")
        logger.info(f"  Last modified: {timestamps['modified_str']}")
        with open(output_file, 'r') as f:
            d = json.load(f)
            conversations['group'] = d.get('group', [])
            conversations['individual'] = d.get('individual', [])

    # Build completion tracking sets - handle both old and new format
    group_completed = set()
    for conv in conversations['group']:
        if args.dataset == QMSUM_DATASET and 'id' in conv and 'start_idx' in conv:
            group_completed.add((conv['id'], conv['start_idx']))
        elif args.dataset == EMPATHETIC_DATASET and 'id' in conv:
            if enable_personality:
                pval = conv.get(personality_field)
                group_completed.add((conv['id'], pval))
            else:
                group_completed.add(conv['id'])

    individual_completed = set()
    for conv in conversations['individual']:
        if args.dataset == QMSUM_DATASET and 'id' in conv and 'start_idx' in conv:
            individual_completed.add((conv['id'], conv['start_idx']))
        elif args.dataset == EMPATHETIC_DATASET and 'id' in conv:
            if enable_personality:
                pval = conv.get(personality_field)
                individual_completed.add((conv['id'], pval))
            else:
                individual_completed.add(conv['id'])

    # Load original data if needed
    original_convs = load_mascot_conversations(input_file)
    if args.max_samples is not None:
        original_convs = original_convs[:args.max_samples]
        logger.info(f"Limiting evaluation to {len(original_convs)} conversations (--max-samples={args.max_samples})")

    # Validate dataset format
    if args.dataset == QMSUM_DATASET:
        if not all('meeting' in conv for conv in original_convs[:5]):
            logger.error(f"Dataset specified as {QMSUM_DATASET!r} but missing 'meeting' field")
            return
    elif args.dataset == EMPATHETIC_DATASET:
        if not all('initial_situation' in conv for conv in original_convs[:5]):
            logger.error(f"Dataset specified as {EMPATHETIC_DATASET!r} but missing 'initial_situation' field")
            return
    logger.info(f"Dataset format validated: {args.dataset}")

    if args.dataset == QMSUM_DATASET:
        expected_group_ids = set((conv['id'], conv['start_idx']) for conv in original_convs)

    elif args.dataset == EMPATHETIC_DATASET:
        if enable_personality:
            expected_group_ids = set((conv['conv_id'], conv.get(personality_field)) for conv in original_convs)
        else:
            expected_group_ids = set(conv['conv_id'] for conv in original_convs)

    # Initialize judge
    judge_model_name = args.judge
    if args.vllm_judge_url:
        judge_model = VLLMServerModel(
            model_name=judge_model_name,
            base_url=args.vllm_judge_url,
            enable_thinking=False,  # Judges produce structured JSON, no thinking needed
        )
    elif "gpt" in judge_model_name.lower():
        judge_model = GPT(model_name=judge_model_name)
    elif "claude" in judge_model_name.lower():
        judge_model = Claude(model_name=judge_model_name)
    elif "gemini" in judge_model_name.lower():
        judge_model = Gemini(model_name=judge_model_name)
    else:
        logger.error(f"Unknown judge model: {judge_model_name}")
        return
    
    if args.dataset == QMSUM_DATASET:
        agents = AGENTS

    elif args.dataset == EMPATHETIC_DATASET:
        agents = EMPATHETIC_AGENTS
    
    
    def save_results(conv):
        with open(output_file, 'w') as f:
            json.dump(conv, f, indent=2)
        logger.info(f"Saved intermediate results to {output_file}")


    # Run individual evaluation if not complete
    if len(conversations['individual']) < len(original_convs):
        logger.info("Running individual evaluation...")
        individual_evaluator = IndividualRubricEvaluator(
            judge_model=judge_model, dataset_name=args.dataset,
            criteria_order=criteria_order_individual)

        # Filter out conversations where ALL turns are already complete
        convs_to_evaluate = []
        for conv in original_convs:
            # Check if all turns for this conversation are complete


            if args.dataset == QMSUM_DATASET:
                if (conv['id'], conv['start_idx']) not in individual_completed:
                    convs_to_evaluate.append(conv)
            elif args.dataset == EMPATHETIC_DATASET:
                if enable_personality:
                    pval = conv.get(personality_field)
                    if (conv['conv_id'], pval) not in individual_completed:
                        convs_to_evaluate.append(conv)
                else:
                    if conv['conv_id'] not in individual_completed:
                        convs_to_evaluate.append(conv)

        logger.info(f"Evaluating {len(convs_to_evaluate)} conversations with incomplete individual evals")

        conversation_batches, _ = prepare_individual_evaluation_tasks(
            args, convs_to_evaluate, args.eval_turns_start, turn_end, agents, args.dataset
        )

        # Phase 1: Build all prompts upfront
        all_prompts = []
        all_meta = []  # (conv_batch, conv, original_responses, cleaned_responses)
        for i, conv_batch in enumerate(conversation_batches):
            conv = convs_to_evaluate[i]
            prompt, orig, clean = individual_evaluator.build_prompt(conv_batch, agents, args.dataset)
            if prompt is None:
                continue
            all_prompts.append(prompt)
            all_meta.append((conv_batch, conv, orig, clean))

        logger.info(f"Sending {len(all_prompts)} individual evaluation prompts to judge in one batch...")

        # Phase 2: Batch generate all prompts at once (with structured output)
        batch_results = _preparse_structured(judge_model.generate_batch(
            prompts=all_prompts,
            temperature=0.2,
            # One list entry per agent turn; long turns need headroom or the
            # output truncates mid-list and the whole conversation fails parsing.
            # 4096 is the Azure gpt-4o deployment's completion-token ceiling.
            max_tokens=4096,
            response_format=individual_evaluator.response_format
        ), judge_model)

        # Phase 3: Parse results with batch retries for parse failures
        # First pass: try to parse all results, collect failures
        responses = list(batch_results)  # mutable copy
        parsed = [None] * len(all_meta)  # will hold result_entry on success
        failed_indices = []

        for batch_idx, (conv_batch, conv, orig, clean) in enumerate(tqdm(all_meta, desc="[Parse] Individual")):
            try:
                results = individual_evaluator.parse_result(responses[batch_idx], conv_batch, orig, clean)
                results_list = []
                for j, result in enumerate(results):
                    result_dict = result.to_dict()
                    result_dict['turn_idx'] = conv_batch[j]['response_idx']
                    result_dict['agent_name'] = conv_batch[j]['agent_name']
                    results_list.append(result_dict)

                    result_str = " | ".join([f"{k[:3]}:{v:.2f}" for k, v in result.scores.items()])
                    result_str += f" | total:{result.total_reward:.2f} | {result.original_response.split('</think>', 1)[-1].strip()[:50]}..."
                    print(f"Turn {result_dict['turn_idx']} [{result_dict['agent_name']}]: {result_str}")
                    print_colored(f"\tAnswer: {result_dict['cleaned_response'][:200]}", "yellow")
                    print_colored(f"\tReasoning: {result.reasoning}", "blue")

                result_entry = {
                    'id': conv['id'] if args.dataset == QMSUM_DATASET else conv['conv_id'],
                    'start_idx': conv['start_idx'] if args.dataset == QMSUM_DATASET else None,
                    'results': results_list,
                    "agent_names": [result['agent_name'] for result in results_list]
                }
                if args.dataset == EMPATHETIC_DATASET and enable_personality:
                    result_entry[personality_field] = conv.get(personality_field)

                parsed[batch_idx] = result_entry
            except Exception as e:
                logger.warning(f"Initial parse failed for conv {batch_idx}: {e}")
                failed_indices.append(batch_idx)

        # Retry rounds: re-submit all failures in one batch per round
        for retry_round in range(individual_evaluator.num_retries):
            if not failed_indices:
                break
            logger.info(f"Individual retry round {retry_round + 1}: re-generating {len(failed_indices)} failed prompts")
            retry_prompts = [all_prompts[i] for i in failed_indices]
            retry_results = _preparse_structured(judge_model.generate_batch(
                prompts=retry_prompts, temperature=0.2, max_tokens=4096,
                response_format=individual_evaluator.response_format
            ), judge_model)
            still_failed = []
            for fi, retry_resp in zip(failed_indices, retry_results):
                conv_batch, conv, orig, clean = all_meta[fi]
                try:
                    results = individual_evaluator.parse_result(retry_resp, conv_batch, orig, clean)
                    results_list = []
                    for j, result in enumerate(results):
                        result_dict = result.to_dict()
                        result_dict['turn_idx'] = conv_batch[j]['response_idx']
                        result_dict['agent_name'] = conv_batch[j]['agent_name']
                        results_list.append(result_dict)
                    result_entry = {
                        'id': conv['id'] if args.dataset == QMSUM_DATASET else conv['conv_id'],
                        'start_idx': conv['start_idx'] if args.dataset == QMSUM_DATASET else None,
                        'results': results_list,
                        "agent_names": [result['agent_name'] for result in results_list]
                    }
                    if args.dataset == EMPATHETIC_DATASET and enable_personality:
                        result_entry[personality_field] = conv.get(personality_field)
                    parsed[fi] = result_entry
                except Exception as e:
                    logger.warning(f"Retry round {retry_round + 1} failed for conv {fi}: {e}")
                    still_failed.append(fi)
            failed_indices = still_failed

        # Per-turn fallback: very long conversations overflow the judge's
        # completion budget when all turns are scored in one prompt, so score
        # each turn with its own single-entry prompt (same rubric).
        if failed_indices:
            logger.info(f"Per-turn fallback for {len(failed_indices)} conversations")
            sub_prompts = []
            sub_meta = []  # (batch_idx, turn_j, sub_batch, orig, clean)
            for fi in failed_indices:
                conv_batch, conv, _, _ = all_meta[fi]
                for j in range(len(conv_batch)):
                    sub_batch = [conv_batch[j]]
                    prompt, orig, clean = individual_evaluator.build_prompt(
                        sub_batch, agents, args.dataset)
                    if prompt is None:
                        continue
                    sub_prompts.append(prompt)
                    sub_meta.append((fi, j, sub_batch, orig, clean))
            sub_results = _preparse_structured(judge_model.generate_batch(
                prompts=sub_prompts, temperature=0.2, max_tokens=4096,
                response_format=individual_evaluator.response_format
            ), judge_model)
            per_conv = {}
            for (fi, j, sub_batch, orig, clean), resp in zip(sub_meta, sub_results):
                try:
                    result = individual_evaluator.parse_result(resp, sub_batch, orig, clean)[0]
                    result_dict = result.to_dict()
                    result_dict['turn_idx'] = sub_batch[0]['response_idx']
                    result_dict['agent_name'] = sub_batch[0]['agent_name']
                    per_conv.setdefault(fi, {})[j] = result_dict
                except Exception as e:
                    logger.warning(f"Per-turn fallback failed for conv {fi} turn {j}: {e}")
            still_failed = []
            for fi in failed_indices:
                conv_batch, conv, _, _ = all_meta[fi]
                turns = per_conv.get(fi, {})
                if len(turns) == len(conv_batch):
                    results_list = [turns[j] for j in range(len(conv_batch))]
                    result_entry = {
                        'id': conv['id'] if args.dataset == QMSUM_DATASET else conv['conv_id'],
                        'start_idx': conv['start_idx'] if args.dataset == QMSUM_DATASET else None,
                        'results': results_list,
                        "agent_names": [r['agent_name'] for r in results_list],
                    }
                    if args.dataset == EMPATHETIC_DATASET and enable_personality:
                        result_entry[personality_field] = conv.get(personality_field)
                    parsed[fi] = result_entry
                else:
                    still_failed.append(fi)
            failed_indices = still_failed

        # Append successful results in original order
        for batch_idx, entry in enumerate(parsed):
            if entry is not None:
                conversations['individual'].append(entry)
            else:
                logger.error(f"All retries failed for conversation {batch_idx}, skipping")
            if (batch_idx + 1) % 20 == 0:
                save_results(conversations)

        save_results(conversations)
        
     # Run group evaluation if not complete
    if group_completed < expected_group_ids:
        logger.info("Running group evaluation...")
        group_evaluator = GroupRubricEvaluator(
            judge_model=judge_model, dataset_name=args.dataset,
            criteria_order=criteria_order_group)

        # Phase 1: Build all group prompts
        pending_group_convs = []
        all_group_prompts = []
        all_group_meta = []
        for conv in original_convs:
            skip = False
            if args.dataset == QMSUM_DATASET:
                skip = (conv['id'], conv['start_idx']) in group_completed
            elif args.dataset == EMPATHETIC_DATASET:
                if enable_personality:
                    pval = conv.get(personality_field)
                    skip = (conv['conv_id'], pval) in group_completed
                else:
                    skip = conv['conv_id'] in group_completed
            if skip:
                continue

            prompt, orig, clean = group_evaluator.build_prompt(conv, agents)
            if prompt is None:
                continue
            pending_group_convs.append(conv)
            all_group_prompts.append(prompt)
            all_group_meta.append((orig, clean))

        logger.info(f"Sending {len(all_group_prompts)} group evaluation prompts to judge in one batch...")

        # Phase 2: Batch generate all prompts at once (with structured output)
        group_batch_results = _preparse_structured(judge_model.generate_batch(
            prompts=all_group_prompts, temperature=0.0, max_tokens=3000,
            response_format=group_evaluator.response_format
        ), judge_model)

        # Phase 3: Parse results with batch retries for parse failures
        group_responses = list(group_batch_results)
        group_parsed = [None] * len(pending_group_convs)  # will hold list of result_dicts
        group_failed_indices = []

        for batch_idx, conv in enumerate(tqdm(pending_group_convs, desc="[Parse] Group")):
            orig, clean = all_group_meta[batch_idx]
            try:
                results = group_evaluator.parse_result(group_responses[batch_idx], conv, orig, clean)
                result_dicts = []
                for j, result in enumerate(results):
                    result_str = " | ".join([f"{k[:3]}:{v:.2f}" for k, v in result.scores.items()])
                    result_str += f" | total:{result.total_reward:.2f} | {result.original_response.split('</think>', 1)[-1].strip()[:50]}..."
                    result_dict = result.to_dict()
                    result_dict['agent_name'] = conv['turns'][j]['agent_name']
                    if args.dataset == EMPATHETIC_DATASET and enable_personality:
                        result_dict[personality_field] = conv.get(personality_field)
                    print(result_str)
                    print_colored(f"\tAnswer: {result_dict.get('cleaned_response', result.original_response.split('</think>', 1)[-1].strip())[:200]}", "yellow")
                    print_colored(f"\tReasoning: {result.reasoning}", "blue")
                    result_dicts.append(result_dict)
                group_parsed[batch_idx] = result_dicts
            except Exception as e:
                traceback.print_exc()
                logger.warning(f"Initial group parse failed for conv {batch_idx}: {e}")
                group_failed_indices.append(batch_idx)

        # Retry rounds: re-submit all failures in one batch per round
        for retry_round in range(group_evaluator.num_retries):
            if not group_failed_indices:
                break
            logger.info(f"Group retry round {retry_round + 1}: re-generating {len(group_failed_indices)} failed prompts")
            retry_prompts = [all_group_prompts[i] for i in group_failed_indices]
            retry_results = _preparse_structured(judge_model.generate_batch(
                prompts=retry_prompts, temperature=0.0, max_tokens=3000,
                response_format=group_evaluator.response_format
            ), judge_model)
            still_failed = []
            for fi, retry_resp in zip(group_failed_indices, retry_results):
                conv = pending_group_convs[fi]
                orig, clean = all_group_meta[fi]
                try:
                    results = group_evaluator.parse_result(retry_resp, conv, orig, clean)
                    result_dicts = []
                    for j, result in enumerate(results):
                        result_dict = result.to_dict()
                        result_dict['agent_name'] = conv['turns'][j]['agent_name']
                        if args.dataset == EMPATHETIC_DATASET and enable_personality:
                            result_dict[personality_field] = conv.get(personality_field)
                        result_dicts.append(result_dict)
                    group_parsed[fi] = result_dicts
                except Exception as e:
                    logger.warning(f"Group retry round {retry_round + 1} failed for conv {fi}: {e}")
                    still_failed.append(fi)
            group_failed_indices = still_failed

        # Append successful results in original order
        for batch_idx, result_dicts in enumerate(group_parsed):
            if result_dicts is not None:
                conversations['group'].extend(result_dicts)
            else:
                logger.error(f"All retries failed for group conversation {batch_idx}, skipping")
            if (batch_idx + 1) % 20 == 0:
                save_results(conversations)
                
    """
    entries, s = [], set()
    for line in conversations['group']:
        if (line['id'], line['start_idx']) in s:
            continue
        entries.append(line)
        s.add((line['id'], line['start_idx']))
    """
                

    save_results(conversations)
    # Validate results
    assert len(conversations['individual']) == len(conversations['group']) == len(original_convs), "Individual and group eval counts must match"
    logger.info(f"✓ Complete! Output: {output_file}")


if __name__ == "__main__":
    main()
