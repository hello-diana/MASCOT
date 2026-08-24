"""Shared infrastructure for the rubric evaluators (Phase-2 Part E).

IndividualRubricEvaluator and GroupRubricEvaluator (src/eval/rubric_evaluator.py)
share their judge-call retry loop, structured-response parsing, rubric-text
assembly, and guided-JSON response-format construction. Those pieces live here;
the public evaluator classes, their prompts, schemas, criterion order, retry
counts, temperatures, and score aggregation are unchanged and stay in
rubric_evaluator.py.
"""

import json
import logging
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    scores: Dict[str, int]
    total_reward: float
    reasoning: str
    id: Optional[str] = None
    start_idx: Optional[int] = None
    weighted_reward: Optional[float] = None
    original_response: Optional[str] = None
    cleaned_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'start_idx': self.start_idx,
            'scores': self.scores,
            'total_reward': self.total_reward,
            'reasoning': self.reasoning,
            'weighted_reward': self.weighted_reward,
            'original_response': self.original_response,
            'cleaned_response': self.cleaned_response
        }


def build_rubric_text(criteria: List[str], order: Optional[List[int]] = None, epilogue: str = "") -> str:
    """Join criterion blocks in the given order, renumbering 1..N.

    order=None reproduces the original rubric text byte-identically.
    """
    if order is None:
        order = list(range(len(criteria)))
    assert sorted(order) == list(range(len(criteria))), \
        f"Invalid criteria order {order} for {len(criteria)} criteria"
    return "\n" + "".join(f"{i + 1}. {criteria[idx]}" for i, idx in enumerate(order)) + epilogue


def _make_response_format(name: str, response_model) -> dict:
    """vLLM guided-JSON response_format for a pydantic response model."""
    return {
        'type': 'json_schema',
        'json_schema': {
            'name': name,
            'strict': False,
            'schema': response_model.model_json_schema()
        }
    }


def _generate_and_parse_with_retries(evaluator, prompt, evaluation_data, orig, clean,
                                     temperature, warn_label, fail_message,
                                     sleep_seconds=0):
    """Shared judge-call + parse retry loop for both rubric evaluators.

    Log text, temperatures, and sleep behavior are parameterized to preserve
    each evaluator's existing behavior exactly.
    """
    for _ in range(evaluator.num_retries):
        try:
            results = evaluator.judge_model.generate_batch(
                prompts=[prompt], temperature=temperature, max_tokens=3000
            )
            return evaluator.parse_result(results[0], evaluation_data, orig, clean)
        except Exception as e:
            logger.warning(f"{warn_label}: {e}")
            traceback.print_exc()
            if sleep_seconds:
                time.sleep(sleep_seconds)
    raise RuntimeError(fail_message)


def _parse_judge_response(response) -> Dict[str, Any]:
    """Parse a judge response into a dict.

    Accepts a dict (structured output already parsed by the caller — the
    former VLLMServerModel ``parsed_json`` path, now made explicit in
    llm_as_a_judge) or a raw string, which goes through the original rubric
    string cascade unchanged.
    """
    if isinstance(response, dict):
        return response

    # Try to extract JSON from response
    response = response.strip()

    # Remove markdown code blocks if present
    if response.startswith("```"):
        lines = response.split("\n")
        # Remove first and last lines (``` markers)
        response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
        if response.startswith("json"):
            response = response[4:].strip()

    # Find JSON object
    start_idx = response.find("{")
    end_idx = response.rfind("}") + 1

    if start_idx != -1 and end_idx > start_idx:
        json_str = response[start_idx:end_idx]
        return json.loads(json_str)
    else:
        raise ValueError(f"Could not find JSON in response: {response[:200]}")
