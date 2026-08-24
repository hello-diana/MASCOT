#!/usr/bin/env python3
"""Transcript and conversation utilities for evaluation scripts.

This module provides shared utility functions for processing meeting transcripts
and formatting conversation histories. These functions are used across multiple
evaluation scripts to avoid code duplication.
"""

from typing import List, Dict
from tqdm import tqdm
from config.persona_config import AGENTS
from src.utils.text import extract_final_response


def segment_transcript(transcript: List[Dict], turns_per_segment: int = 100) -> List[List[Dict]]:
    """Segment a meeting transcript into chunks of fixed size.

    Args:
        transcript: List of transcript turns, each with 'speaker' and 'text' keys
        turns_per_segment: Number of turns per segment (default: 100)

    Returns:
        List of transcript segments, each containing turns_per_segment turns
    """
    segments = []
    for i in range(0, len(transcript), turns_per_segment):
        segment = transcript[i:i + turns_per_segment]
        segments.append(segment)
    return segments


def format_transcript_segment(segment: List[Dict]) -> str:
    """Convert transcript segment to formatted string.

    Args:
        segment: List of transcript turns with 'speaker' and 'text' keys

    Returns:
        Formatted string with each turn on a new line
    """
    return "\n".join([f"{turn['speaker']}: {turn['text']}" for turn in segment])


def format_agent_personas() -> str:
    """Format all agent personas for evaluation context.

    Returns:
        Formatted string with all agent personas from AGENTS config
    """
    personas = []
    for agent_key, persona_spec in AGENTS.items():
        personas.append(f"**{persona_spec.name}**:\n{persona_spec.description}")
    return "\n\n".join(personas)


def create_meeting_contexts(dataset, turns_per_segment: int = 100) -> List[Dict]:
    """Segment meeting transcripts into fixed-size contexts.

    This function chunks meeting transcripts into segments of a specified size,
    creating meeting contexts that can be processed in batch. Each context
    includes the meeting ID, segment indices, and formatted transcript text.

    Args:
        dataset: HuggingFace dataset with meeting transcripts
        turns_per_segment: Number of turns per segment (default: 100)

    Returns:
        List of meeting context dicts with structure:
        {
            'id': str,           # Meeting ID
            'start_idx': int,    # Starting turn index
            'end_idx': int,      # Ending turn index
            'meeting_context': str  # Formatted transcript segment
        }
    """
    meeting_contexts = []

    for meeting in tqdm(dataset, desc="Preparing meeting contexts"):
        transcript = meeting["meeting_transcripts"]

        for start_idx in range(0, len(transcript), turns_per_segment):
            end_idx = start_idx + turns_per_segment
            if end_idx > len(transcript):
                continue

            # Format context segment
            context_turns = transcript[start_idx:end_idx]
            meeting_context = "\n".join([
                f"{turn['speaker']}: {turn['content']}"
                for turn in context_turns
            ])

            meeting_contexts.append({
                "id": meeting["id"],
                "start_idx": start_idx,
                "end_idx": end_idx,
                "meeting_context": meeting_context
            })

    return meeting_contexts


def format_conversation_history(turns: List[Dict], name_key: str = 'agent_name') -> str:
    """Format conversation history, stripping <think></think> tags.

    This function formats a list of conversation turns into a string, removing
    thinking tags if present. It supports flexible key names for agent names
    to work with different data structures.

    Args:
        turns: List of turn dicts with agent name and response
        name_key: Key to use for agent name ('agent_name' or 'name')

    Returns:
        Formatted conversation string with agent names and responses

    Examples:
        >>> turns = [{'agent_name': 'Alice', 'response': '<think>...</think>Hello'}]
        >>> format_conversation_history(turns)
        'Alice: Hello\\n'

        >>> turns = [{'name': 'Bob', 'response': 'Hi there'}]
        >>> format_conversation_history(turns, name_key='name')
        'Bob: Hi there\\n'
    """
    conversation_history = ""
    for turn in turns:
        # Strip thinking tags if present
        clean_response = extract_final_response(turn['response'])
        if turn.get('turn_type') == 'user':
            # Simulated-user turns carry no agent name.
            conversation_history += f"[User]: {clean_response}\n"
        else:
            conversation_history += f"[{turn[name_key]} agent]: {clean_response}\n"
    return conversation_history


def format_conversation_history_empathetic(turns: List[Dict]) -> str:
    """Format conversation history with interleaved agent and user turns.

    Args:
        turns: List of turn dicts (both agent and user).  Each must have
            ``response`` and ``turn_type`` ('agent' or 'user').  Agent turns
            must also have ``agent_name``.

    Returns:
        Formatted conversation string
    """
    conversation_history = ""
    for turn in turns:
        clean_response = extract_final_response(turn['response'])
        if turn.get('turn_type') == 'agent':
            conversation_history += f"[{turn['agent_name']}]: {clean_response}\n"
        elif turn.get('turn_type') == 'user':
            conversation_history += f"[User]: {clean_response}\n"
    return conversation_history
