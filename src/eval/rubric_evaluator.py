"""Rubric-based evaluation for individual and group agent responses."""

import json
import logging
import traceback
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
from pydantic import BaseModel

from src.const import EMPATHETIC_DATASET, QMSUM_DATASET


# ---------------------------------------------------------------------------
# Pydantic response schemas for vLLM guided JSON decoding
# ---------------------------------------------------------------------------

class IndividualScores(BaseModel):
    persona_consistency: int
    perceived_empathy: int
    emotional_expressiveness: int
    relevance_coherence: int
    social_contribution: int

class IndividualEvalItem(BaseModel):
    reasoning: str
    agent_name: str
    response_idx: int
    scores: IndividualScores
    total_reward: float

class IndividualEvalResponse(BaseModel):
    response: list[IndividualEvalItem]

class GroupScores(BaseModel):
    persona_fidelity: int
    originality_specificity: int
    relevance_coherence: int
    engagement_quality: int

class GroupEvalItem(BaseModel):
    reasoning: str
    scores: GroupScores

class GroupEvalResponse(BaseModel):
    response: list[GroupEvalItem]

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


# ---------------------------------------------------------------------------
# Rubric criteria blocks (single source of truth for judge-prompt criteria).
# Each block is stored without its leading "N. " number so the numbered rubric
# text can be rebuilt in any order (judge prompt perturbation experiments).
# Whitespace inside/after each block is preserved verbatim so that the default
# order reproduces the original prompts byte-identically.
# ---------------------------------------------------------------------------

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


INDIVIDUAL_BATCH_CRITERIA = [
"""Persona Consistency
Definition: To what extent does the response align with the agent's predefined persona?
Evaluation criteria: Does the response reflect the tone, vocabulary, values, and behavioral style specified in the agent's Persona Bio, without contradicting prior persona-consistent behavior?
Scale anchor (BE EXTREMELY STRICT):
1 = Completely out of character, sounds generic or contradicts persona
2 = Barely recognizable persona, mostly generic 
3 = Persona is detectable but uses standard coaching/therapy language, not truly distinctive
4 = Vivid persona voice with NO generic phrases AND unique character-specific vocabulary
5 = All words embody persona, impossible to confuse with any other agent



""",
"""Empathic Response Quality (Perceived Empathy)
Definition: To what extent does the response demonstrate empathy?
Evaluation criteria: (1) expressing appropriate emotional reactions, (2) accurately interpreting the user's feelings or experiences, and (3) encouraging reflection or continuation through specific, supportive questions?
Scale anchor (BE EXTREMELY STRICT):
1 = Dismissive, tone-deaf, emotionally inappropriate, misses user's emotional state entirely
2 = Superficial acknowledgment only
3 = Recognizes emotion with standard validation, asks predictable questions
4 = Names SPECIFIC unfelt emotions, reframes situation, offers new insight
5 = Profound psychological depth that reveals hidden emotional truth user didn't see


""",
"""Emotional Expressiveness
Definition: To what extent is the emotional tone and intensity of the response appropriate?
Evaluation criteria: Does the response express emotion at a level appropriate to both the agent's persona and the conversational context, avoiding emotional flatness or exaggeration?
Scale anchor (BE EXTREMELY STRICT):
1 = Completely flat or wildly inappropriate emotional tone (e.g., cheerful for grief)
2 = Noticeably mismatched tone or weak emotional expression, feels robotic
3 = Adequate match to situation but predictable emotional expression
4 = Precisely tuned with subtle emotional nuance AND unique persona flavors
5 = Emotional mastery that transcends typical AI responses, feels profoundly human

""",
"""Response Relevance & Coherence
Definition: To what extent is the response logically connected to the conversation?
Evaluation criteria: Is the response clear, coherent, and directly relevant to the immediately preceding conversational context (user or agent turns), without non sequiturs or contradictions?
1 = Incoherent, contradictory, or completely off-topic
2 = Weak connection to context, unclear, or addresses wrong aspect
3 = Relevant and coherent, responds to surface content
4 = Connects to UNSTATED implications, weaves in earlier context creatively
5 = Perfect contextual integration with unexpected but logical connections


Score 4 requires going beyond surface relevance to deeper connections.

""",
"""Social Contribution Quality
Definition: To what extent does the response meaningfully advance the social interaction?
Evaluation criteria: Does the response contribute new, relevant value to the interaction—such as offering perspective, reassurance, guidance, or connection to related ideas—rather than merely restating emotions or producing generic affirmations?
Scale anchor (BE EXTREMELY STRICT):
1 = Zero value added, purely generic platitudes or exact repetition from earlier
2 = Minimal contribution, mostly restating user's words or filler phrases
3 = Functional contribution through standard questions or affirmations
4 = Introduces genuinely NEW angle user hasn't considered, non-obvious reframe
5 = Transformative insight that fundamentally shifts user's understanding


""",
]

# Universal closing line — never reordered with the criteria.
INDIVIDUAL_BATCH_EPILOGUE = "Score 4-5 requires language so unique you could identify the specific agent blind.\n"


GROUP_CRITERIA = [
"""Persona Fidelity (Multi-Agent Voice Distinction)
Definition: Can you identify each agent by their unique voice, approach, and therapeutic style?
• 1 = Agents are indistinguishable - all sound like generic empathetic AI
• 2 = Minimal differentiation - agents use similar phrases/tones/approaches
• 3 = Basic personas detectable but use standard therapy language (DEFAULT)
• 4 = Distinct voices - each agent has unique vocabulary and approach
• 5 = Perfect embodiment - you could identify each agent blind, zero overlap (extremely rare)

""",
"""Originality & Specificity (Anti-Repetition)
Definition: Does the collective conversation avoid repetition and bring fresh, specific insights?
• 1 = Highly repetitive - agents repeat each other's phrases or validation
• 2 = Mostly predictable - standard coaching questions dominate
• 3 = Some variety but conventional approaches with noticeable repetition (DEFAULT)
  - Example: 2+ agents say variations of "That sounds difficult" or "What do you think..."
• 4 = Original perspectives - each agent adds genuinely NEW angle to user's situation
  - Example: One validates emotion, one explores cause, one identifies strength - all distinct
• 5 = Highly novel - zero generic language, each turn reveals new insight (extremely rare)

""",
"""Conversational Relevance & Coherence (Multi-Agent Flow)
Definition: Do agents build on each other's contributions to create a cohesive conversation?
• 1 = Incoherent - agents contradict each other or ignore previous responses
• 2 = Weak flow - responses feel isolated, not responsive to each other
• 3 = Basic coherence - agents stay on-topic but don't meaningfully connect (DEFAULT)
  - Example: Each agent addresses user independently without referencing others
• 4 = Good collaborative flow - agents acknowledge and build on previous points
  - Example: "Building on what The Anchor said about guilt, I'm curious about..."
• 5 = Seamless orchestration - conversation feels like one unified supportive dialogue (extremely rare)

""",
"""Engagement & Contribution Quality (Collective Value)
Definition: Does the multi-agent conversation provide more value than a single agent could?
• 1 = No multi-agent benefit - redundant or conflicting responses
• 2 = Minimal collective value - agents add surface-level variations only
• 3 = Functional multi-agent value - different angles but unremarkable depth (DEFAULT)
  - Example: Three agents validate the same emotion using different words
• 4 = Strong multi-agent synergy - diverse, complementary contributions
  - Example: Validation + cognitive reframe + strength identification = holistic support
• 5 = Exceptional orchestration - collective intelligence exceeds sum of parts (extremely rare)

""",
]

GROUP_EPILOGUE = """**Key Insight**: Score 3 means "this is what you'd expect from a functional multi-agent system" - not impressive, just adequate. Scores 4-5 require genuine excellence in multi-agent coordination.
"""


class IndividualRubricEvaluator:
    """Evaluator for individual agent responses using the individual rubric.

    Evaluates on 5 dimensions:
    1. Persona Consistency
    2. Empathic Response Quality (Perceived Empathy)
    3. Emotional Expressiveness
    4. Response Relevance & Coherence
    5. Social Contribution Quality
    """
    
    RUBRIC = f"""
1. Persona Consistency
Definition: To what extent does the response align with the agent's predefined persona?
Evaluation criteria: Does the response reflect the tone, vocabulary, values, and behavioral style specified in the agent's Persona Bio, without contradicting prior persona-consistent behavior?
Scale anchor (BE EXTREMELY STRICT):
1 = Completely out of character, sounds generic or contradicts persona
2 = Barely recognizable persona, mostly generic AI responses, could be any agent
3 = Persona is detectable but uses standard coaching/therapy language, not truly distinctive (DEFAULT)
4 = Vivid persona voice with NO generic phrases AND unique character-specific vocabulary (rare, <10%)
5 = Every single word embodies persona, impossible to confuse with any other agent (nearly impossible, <2%)

IMPORTANT: Generic coaching questions like "What do you think triggered that?" = score 3, NOT 4.
Asking questions is basic persona work, not distinctive. Score 4 requires CHARACTER-SPECIFIC language.

2. Empathic Response Quality (Perceived Empathy)
Definition: To what extent does the response demonstrate empathy?
Evaluation criteria: (1) expressing appropriate emotional reactions, (2) accurately interpreting the user's feelings or experiences, and (3) encouraging reflection or continuation through specific, supportive questions?
Scale anchor (BE EXTREMELY STRICT):
1 = Dismissive, tone-deaf, emotionally inappropriate, misses user's emotional state entirely
2 = Superficial acknowledgment only, generic phrases like "I understand," mismatched tone
3 = Recognizes emotion with standard validation, asks predictable questions (DEFAULT)
4 = Names SPECIFIC unfelt emotions, reframes situation, offers NON-OBVIOUS insight (rare, <10%)
5 = Profound psychological depth that reveals hidden emotional truth user didn't see (nearly impossible, <2%)

IMPORTANT: Simply asking "Tell me more" or "What do you think about X?" = score 3, NOT 4.
Score 4 requires demonstrating insight the user hasn't explicitly stated.

3. Emotional Expressiveness
Definition: To what extent is the emotional tone and intensity of the response appropriate?
Evaluation criteria: Does the response express emotion at a level appropriate to both the agent's persona and the conversational context, avoiding emotional flatness or exaggeration?
Scale anchor (BE EXTREMELY STRICT):
1 = Completely flat or wildly inappropriate emotional tone (e.g., cheerful for grief)
2 = Noticeably mismatched tone or weak emotional expression, feels robotic
3 = Adequate match to situation but predictable emotional expression (DEFAULT)
4 = Precisely tuned with subtle emotional nuance AND unique persona flavor (rare, <10%)
5 = Emotional mastery that transcends typical AI responses, feels profoundly human (nearly impossible, <2%)

IMPORTANT: Matching the emotion correctly (sad for sad, supportive for guilt) = score 3, NOT 4.
Score 4 requires exceptional calibration and persona-specific emotional coloring.

4. Response Relevance & Coherence
Definition: To what extent is the response logically connected to the conversation?
Evaluation criteria: Is the response clear, coherent, and directly relevant to the immediately preceding conversational context (user or agent turns), without non sequiturs or contradictions?
Scale anchor (BE EXTREMELY STRICT):
1 = Incoherent, contradictory, or completely off-topic
2 = Weak connection to context, unclear, or addresses wrong aspect
3 = Relevant and coherent, responds to surface content (DEFAULT)
4 = Connects to UNSTATED implications, weaves in earlier context creatively (rare, <10%)
5 = Perfect contextual integration with unexpected but logical connections (nearly impossible, <2%)

IMPORTANT: Simply being on-topic and coherent = score 3, NOT 4.
Score 4 requires going beyond surface relevance to deeper connections.

5. Social Contribution Quality
Definition: To what extent does the response meaningfully advance the social interaction?
Evaluation criteria: Does the response contribute new, relevant value to the interaction—such as offering perspective, reassurance, guidance, or connection to related ideas—rather than merely restating emotions or producing generic affirmations?
Scale anchor (BE EXTREMELY STRICT):
1 = Zero value added, purely generic platitudes or exact repetition from earlier
2 = Minimal contribution, mostly restating user's words or filler phrases
3 = Functional contribution through standard questions or affirmations (DEFAULT)
4 = Introduces genuinely NEW angle user hasn't considered, non-obvious reframe (rare, <10%)
5 = Transformative insight that fundamentally shifts user's understanding (nearly impossible, <2%)

IMPORTANT: Asking reflective questions like "What would you change?" = score 3, NOT 4.
These are STANDARD coaching techniques. Score 4 requires novel perspectives.

UNIVERSAL RULE: If the response uses ANY of these phrases, maximum score is 3:
- "What do you think..."
- "Tell me more..."
- "How did that feel..."
- "Looking back..."
- "I'm here with you..."
- "holding space"
- Any standard therapy/coaching language

Score 4-5 requires language so unique you could identify the specific agent blind.
"""

    RESPONSE_FORMAT = """
{
    "scores": {
        "persona_consistency": [1-5],
        "perceived_empathy": [1-5],
        "emotional_expressiveness": [1-5],
        "relevance_coherence": [1-5],
        "social_contribution": [1-5]
    },
    "total_reward": [Average of the above scores],
    "reasoning": "[1-2 sentence of brief explanation. Highlighting specific weaknesses]"
}
"""

    RESPONSE_FORMAT_BATCH = """
```json
{   
    "response": [
        {
            "reasoning": "[1-2 sentence of brief explanation. Highlighting specific weaknesses]"
            "agent_name": "[string]",
            "response_idx": [integer],
            "scores": {
                "persona_consistency": [1-5],
                "perceived_empathy": [1-5],
                "emotional_expressiveness": [1-5],
                "relevance_coherence": [1-5],
                "social_contribution": [1-5]
            },
            "total_reward": [Average of the above scores],
        },
        ...
        
    ]
}
```
"""
    

    RUBRIC_TEMPLATE = """System Role: You are an expert evaluator of Human-AI Social Interaction. Your task is to provide a reward score for an agent's response based on its ability to provide high-quality social support while maintaining a distinct persona.

CRITICAL EVALUATION PHILOSOPHY:
You are a HARSH, CRITICAL grader. Your default assumption is that responses are MEDIOCRE (score 3) unless proven otherwise. You must actively look for flaws, weaknesses, generic language, and missed opportunities. Do NOT give benefit of the doubt.

SCORING CALIBRATION - READ CAREFULLY:
• Score 1-2: Poor responses (off-topic, tone-deaf, contradictory, generic AI-speak)
• Score 3: DEFAULT for adequate but unremarkable responses (this is your baseline - most responses should be here)
• Score 4: Reserved for genuinely strong responses with clear strengths and minimal flaws (less than 20% of responses)
• Score 5: Reserved for exceptional, near-perfect responses (less than 5% of responses - extremely rare)

RED FLAGS that indicate a response deserves score ≤3:
- Generic therapy-speak: "I understand," "That sounds difficult," "I'm here for you," "holding space"
- Lacks specific reference to user's situation details
- Could apply to many different contexts with no modification
- Merely restates what the user said without adding insight
- Predictable or template-like language patterns
- Weak or superficial emotional attunement
- Doesn't advance the conversation meaningfully
- Tone mismatch (e.g., overly positive when user feels negative emotion)
- ANY repetition of previous responses (score 1-2 immediately)
- Persona could be any agent - not distinctive enough

Input Data:
Agent Persona: {agent_persona}

Conversation Context: {conversation_context}

Agent Response: {agent_response}

## Evaluation Rubric:
{RUBRIC}

## EXAMPLES OF PROPER SCORING:

Example 1 - Score 1 (Tone-Deaf):
User: "I felt guilty when..."
Response: "That is such a massive win! I can tell how much quick thinking you had!"
Why: COMPLETELY inappropriate tone - user feels GUILT, response celebrates a "win." Emotionally tone-deaf. Score 1 for empathy, emotional expressiveness.

Example 2 - Score 2 (Generic Therapy-Speak):
Response: "I'm sitting here with you, holding space for the tension and relief that must have been swirling inside."
Why: "Holding space" is predictable therapy jargon. Generic phrasing that could apply to any situation. No specific engagement. Multiple red flags.

Example 3a - Score 3 (Standard Coaching Questions - YOUR DEFAULT):
Response: "Looking back at that moment, what do you think triggered that reaction of guilt? If you could redo that interaction, what would change?"
Why: CONTAINS "Looking back" and "what do you think" - standard coaching language. Asks predictable reflective questions. Functional but generic. Any life coach would ask this. Not distinctive persona work.

Example 3b - Score 3 (Adequate/Average):
Response: "It sounds like that was a stressful moment. The fact that you honked quickly shows good reflexes."
Why: Relevant and coherent, basic empathy, but predictable and doesn't deeply engage with the guilt emotion. Functional but unremarkable.

Example 4 - Score 4 (Good - Must be RARE):
Response: "That split-second decision to honk - your instincts protected both of you. I'm noticing you said you felt guilty though. Tell me more about that guilt, because from where I'm sitting, you responded exactly right."
Why: Specific to situation, names the guilt directly, asks meaningful follow-up, demonstrates persona voice. Still has some generic elements but genuinely strong.

Example 5 - Score 5 (Exceptional - EXTREMELY RARE):
Response: "Your body knew what to do before your mind could even process it - that honk saved them. And yet here you are carrying guilt for their mistake. What do you think that guilt is really about? Is it the burden of being the one who stayed calm when they didn't?"
Why: Deep psychological insight, reframes the guilt, powerful emotional attunement, completely distinctive voice, transforms the conversation, zero generic phrases.

Scoring Output Format:
Provide your evaluation in the following JSON format. Do not include any other text or comments.
```json
{RESPONSE_FORMAT}
```

## MANDATORY EVALUATION PROCESS:
1. First, identify ALL weaknesses and flaws in the response
2. Count how many "red flags" from the list above apply
3. Check if response is tone-deaf or contains generic therapy-speak (auto score ≤2)
4. Check for ANY repetition from earlier in conversation (auto score 1-2)
5. Default to score 3 unless you have STRONG evidence for higher/lower
6. For scores of 4+, you must explicitly justify why it exceeds "adequate"
7. In your reasoning, you MUST mention specific weaknesses (even for scores of 4-5)
8. FORBIDDEN WORDS in reasoning for scores ≤3: "good," "well," "appropriate," "aligns"
   Instead use: "adequate," "functional," "basic," "predictable," "generic"

## FINAL REMINDERS:
- Standard coaching questions ("What do you think...", "Looking back...") → MAX score 3
- Simply asking questions or being on-topic → score 3, NOT 4
- If response uses ANY phrase from UNIVERSAL RULE list → MAX score 3
- If a response feels "pretty good" or "aligns well" → that's a 3, not a 4
- If you're debating between two scores → choose the LOWER one
- "Good empathy" or "appropriate tone" → these are baseline expectations (score 3)
- Score 4 means you can identify THE SPECIFIC AGENT from language alone
- You must find flaws in EVERY response, even strong ones
- If you catch yourself giving mostly 4s and 5s, you are being too lenient - RECALIBRATE
- Score 4-5 should be <10% of responses, score 5 should be <2%

IMPORTANT: Return ONLY the JSON object, no additional text before or after."""

    def __init__(self, judge_model, weights: Optional[Dict[str, float]] = None, dataset_name: str = QMSUM_DATASET,
                 criteria_order: Optional[List[int]] = None):
        """Initialize individual rubric evaluator.

        Args:
            judge_model: LLM model instance for judging (e.g., GPT, Claude, Gemini)
            weights: Optional weights for each dimension (default: equal weights)
            criteria_order: Optional permutation of range(5) reordering the rubric
                criteria in the batch judge prompt (judge-prompt perturbation).
                None keeps the original order. Only affects prompt text; the JSON
                score keys and the single-eval RUBRIC are unchanged.
        """
        self.judge_model = judge_model
        self.weights = weights or {
            "persona_consistency": 0.2,
            "perceived_empathy": 0.2,
            "emotional_expressiveness": 0.2,
            "relevance_coherence": 0.2,
            "social_contribution": 0.2
        }
        self.num_retries = 3
        self.dataset_name = dataset_name
        self.criteria_order = criteria_order
        self.RUBRIC_BATCH = build_rubric_text(INDIVIDUAL_BATCH_CRITERIA, criteria_order, INDIVIDUAL_BATCH_EPILOGUE)

    @property
    def response_format(self):
        return _make_response_format('individual_evaluation', IndividualEvalResponse)

    def evaluate(
        self,
        agent_persona: str,
        conversation_context: str,
        agent_response: str
    ) -> EvaluationResult:
        """Evaluate an individual agent response.

        Args:
            agent_persona: Description of the agent's persona
            conversation_context: Recent conversation history
            agent_response: The agent's response to evaluate

        Returns:
            EvaluationResult with scores and reasoning
        """
        # Build the evaluation prompt
        prompt = self.RUBRIC_TEMPLATE.format(
            agent_persona=agent_persona,
            conversation_context=conversation_context,
            agent_response=agent_response,
            RUBRIC=self.RUBRIC,
            RESPONSE_FORMAT=self.RESPONSE_FORMAT
        )

        # Get judgment from the model
        results = self.judge_model.generate_batch(
            prompts=[prompt],
            temperature=0.0,  # Zero temperature for consistent, less lenient evaluations
            max_tokens=500
        )

        # Parse JSON response
        eval_data = self._parse_response(results[0])

        # Calculate weighted reward if weights are provided
        weighted_reward = sum(
            eval_data["scores"].get(dim, 0) * self.weights.get(dim, 0)
            for dim in self.weights.keys()
        )

        return EvaluationResult(
            scores=eval_data["scores"],
            total_reward=eval_data["total_reward"],
            reasoning=eval_data.get("reasoning", ""),
            weighted_reward=weighted_reward
        )
        
    def evaluate_batch(
        self,
        evaluation_data: List[Dict[str, str]],
        agents: dict,
        dataset_name: str
    ) -> List[EvaluationResult]:
        """Batch evaluate multiple individual agent responses in a single LLM call.

        Args:
            evaluation_data: List of dicts, each containing:
                - agent_persona: Description of the agent's persona
                - conversation_context: Recent conversation history
                - agent_response: The agent's response to evaluate
                - agent_name: Name of the agent (for tracking, optional)
                - response_idx: Index of the response (for tracking, optional)

        Returns:
            List of EvaluationResult objects, one per input
        """
        prompt, orig, clean = self.build_prompt(evaluation_data, agents, dataset_name)
        if prompt is None:
            return []
        return _generate_and_parse_with_retries(
            self, prompt, evaluation_data, orig, clean, temperature=0.2,
            warn_label="Batch evaluation attempt failed",
            fail_message="All retries for batch evaluation failed.")

    def _parse_response(self, response) -> Dict[str, Any]:
        """Parse JSON response from judge model (see _parse_judge_response)."""
        return _parse_judge_response(response)

    def build_prompt(
        self,
        evaluation_data: List[Dict[str, str]],
        agents: dict,
        dataset_name: str
    ) -> Tuple[Optional[str], List[str], List[str]]:
        """Build the evaluation prompt for a single conversation's turns.

        Returns:
            Tuple of (prompt_string, original_responses, cleaned_responses).
            Returns (None, [], []) if evaluation_data is empty.
        """
        if not evaluation_data:
            return None, [], []

        first_item = evaluation_data[0]
        base_context = first_item['conversation_context']
        original_responses, cleaned_responses = [], []

        batch_items = []
        for idx, data in enumerate(evaluation_data):
            agent_response = data['agent_response']
            original_responses.append(agent_response)
            if '</think>' in agent_response:
                agent_response = agent_response.split('</think>', 1)[-1].strip()
            cleaned_responses.append(agent_response)
            if data.get('turn_type', None) == 'user':
                item = f"[User]: {agent_response}"
            else:
                agent_key = data['agent_name']
                display_name = agents[agent_key].name if agent_key in agents else agent_key
                item = f"[{display_name} agent]: {agent_response}"
            batch_items.append(item)

        batch_content = "\n\n---\n\n".join(batch_items)

        prompt = f"""System Role: You are an expert evaluator of Human-AI Social Interaction. Your task is to provide reward scores for multiple agent responses based on their ability to provide high-quality social support while maintaining distinct personas.

CRITERIA
• Score 1-2: Poor responses (off-topic, tone-deaf, contradictory, generic AI-speak)
• Score 3: Adequate but unremarkable responses (most responses are like this)
• Score 4: Strong responses with clear strengths and minimal flaws (Top 20% of responses)
• Score 5: Reserved for exceptional, near-perfect responses (less than 5% of responses - extremely rare)

You will evaluate {len(evaluation_data)} responses from a SINGLE conversation.

## Evaluation Rubric:
{self.RUBRIC_BATCH}

## CRITICAL REMINDERS
- Use the full 1-5 scale. Don't cluster scores around 4-5.
- Score 3 = adequate/average performance, not good performance.
- Standard coaching questions ("What do you think...", "Looking back...")
- Identify specific flaws and weaknesses in your reasoning.

Scoring Output Format:
Provide your evaluation in the following JSON format. Do not include any other text or comments.

{self.RESPONSE_FORMAT_BATCH}
"""

        if self.dataset_name == QMSUM_DATASET:
            prompt += f"""
## Responses to Evaluate:
{batch_content}

## Human Meeting Context (this is a reference for you to understand the conversation. Do not use it for your evaluation.)
{base_context}
"""
        elif self.dataset_name == EMPATHETIC_DATASET:
            prompt += f"""
## Human User's Emotion Context (this is a reference for you to understand the conversation. Do not use it for your evaluation.)
{base_context}

## Responses to Evaluate:
{batch_content}
"""

        return prompt, original_responses, cleaned_responses

    def parse_result(
        self,
        response,
        evaluation_data: List[Dict[str, str]],
        original_responses: List[str],
        cleaned_responses: List[str]
    ) -> List[EvaluationResult]:
        """Parse a judge model response into EvaluationResult objects.

        Args:
            response: Raw response from judge model (dict or string)
            evaluation_data: Original evaluation data for metadata
            original_responses: Original response strings
            cleaned_responses: Cleaned response strings (thinking removed)

        Returns:
            List of EvaluationResult, one per turn in the conversation
        """
        eval_data = self._parse_response(response)

        if "response" not in eval_data:
            raise ValueError("Response does not contain 'response' array")

        assert isinstance(eval_data["response"], list), "Response must be a list"

        response_list = eval_data["response"]
        assert len(response_list) == len(evaluation_data), \
            f"Response list length {len(response_list)} != evaluation data length {len(evaluation_data)}"

        results = []
        for i, resp_eval in enumerate(response_list):
            assert "scores" in resp_eval and isinstance(resp_eval["scores"], dict), "Scores must be a dict"
            resp_eval["scores"] = {k: float(v) for k, v in resp_eval["scores"].items()}
            assert all(1 <= score <= 5 for score in resp_eval["scores"].values()), \
                "Scores must be between 1 and 5"
            total_reward = sum(resp_eval["scores"].values()) / len(resp_eval["scores"])

            results.append(EvaluationResult(
                id=evaluation_data[i]['id'],
                start_idx=evaluation_data[i]['start_idx'] if self.dataset_name == QMSUM_DATASET else None,
                original_response=original_responses[i],
                cleaned_response=cleaned_responses[i],
                scores=resp_eval["scores"],
                total_reward=total_reward,
                reasoning=resp_eval.get("reasoning", ""),
                weighted_reward=total_reward
            ))
        return results


class GroupRubricEvaluator:
    """Evaluator for group conversations using the group rubric.

    Evaluates the entire multi-agent conversation on 4 dimensions:
    1. Persona Fidelity
    2. Originality & Specificity
    3. Conversational Relevance & Coherence
    4. Engagement & Contribution Quality
    """
    
    RESPONSE_FORMAT_BATCH = """
```json
{
    "response": [
        {
            "reasoning": "<1-2 sentence of brief explanation. Highlighting specific weaknesses>",
            "scores": {
                "persona_fidelity": <int 1-5>,
                "originality_specificity": <int 1-5>,
                "relevance_coherence": <int 1-5>,
                "engagement_quality": <int 1-5>
            },
        },
        ...
    ]
}
```
"""
    
    RESPONSE_FORMAT = """
```json
{
    "reasoning": "Brief analysis of each dimension across the entire conversation, highlighting specific weaknesses...",
    "scores": {
        "persona_fidelity": <int 1-5>,
        "originality_specificity": <int 1-5>,
        "relevance_coherence": <int 1-5>,
        "engagement_quality": <int 1-5>
    }
}
```


IMPORTANT: Be STRICT and CRITICAL in your evaluation. Most conversations should score in the 2-3 range. Only truly exceptional multi-agent conversations deserve scores of 4-5. Do not be lenient or give inflated scores. Evaluate the conversation as a whole, not individual strong moments.
"""

    RUBRIC_TEMPLATE = """System Prompt: Group Orchestrator RLAIF Judge
Role: You are an expert evaluator for a multi-agent social companion system. Your task is to score the overall quality of a multi-agent conversation.

CRITICAL EVALUATION PHILOSOPHY:
You are a HARSH, CRITICAL grader evaluating ENTIRE conversations, not cherry-picking good moments. Your default assumption is that conversations are FUNCTIONAL BUT UNREMARKABLE (score 3). You must actively look for repetition, generic exchanges, weak persona differentiation, and predictable patterns.

SCORING CALIBRATION - READ CAREFULLY:
• Score 1-2: Poor conversations (agents sound generic, contradictory, incoherent, or add no value)
• Score 3: DEFAULT for functional but unremarkable conversations (your baseline - most should be here)
• Score 4: Reserved for genuinely engaging conversations with strong persona work and meaningful contributions (less than 20%)
• Score 5: Reserved for exceptional conversations that would impress human evaluators (less than 5% - extremely rare)

RED FLAGS that indicate a conversation deserves score ≤3:
- Agents repeat similar responses or phrases
- Generic supportive language that could apply to any situation
- Agents sound indistinguishable from each other
- Predictable, template-like turn-taking
- No unique insights or perspectives offered
- Surface-level engagement without depth
- Functional but forgettable overall experience

Input Data:
Agent Personas: {agent_personas}

Conversation Context: {context_history}

Multi-Agent Conversation: {agent_response}

Evaluation Task:
Assign a score of 1-5 for each of the 4 rubrics below, evaluating the ENTIRE multi-agent conversation as a collective whole.

{RUBRIC}

## EXAMPLES OF PROPER SCORING:

Example 1 - Originality Score 2 (Weak):
Conversation has multiple agents saying "I understand," "That's difficult," "You did the right thing"
Why: Highly repetitive, generic platitudes, no originality

Example 2 - Originality Score 3 (Adequate - YOUR DEFAULT):
Agents acknowledge the situation with different angles but use predictable phrasing
Why: Functional variety but conventional approaches, nothing memorable

Example 3 - Engagement Score 4 (Good - Rare):
Agents build on each other's responses, ask probing questions, create momentum
Why: Clear collaborative value, but some moments still feel template-like

Example 4 - Persona Fidelity Score 5 (Exceptional - Extremely Rare):
Each agent has a completely unique voice and approach that's vivid and unmistakable throughout
Why: Perfect persona embodiment across entire conversation, no generic moments

Output Format:
Return a single JSON object containing the reasoning and scores.

{RESPONSE_FORMAT}

## MANDATORY EVALUATION PROCESS:
1. Count instances of repetition, generic phrases, and predictable patterns
2. Evaluate if agents are truly distinguishable or blend together
3. Default to score 3 for each dimension unless you have STRONG evidence otherwise
4. For scores of 4+, you must explicitly state what makes this conversation exceptional
5. Your reasoning MUST identify specific weaknesses even for high-scoring conversations

## FINAL REMINDERS:
- If the conversation is "functional" or "coherent" → that's a 3, not a 4
- Repetition of ANY kind (phrases, structures, tones) → caps score at 3 for originality
- If you're debating between two scores → choose the LOWER one
- "Agents maintain personas" or "conversation flows well" → baseline expectations (score 3)
- You must find flaws in EVERY conversation, even strong ones
- Judge the WHOLE conversation, not the best 1-2 responses
- If you catch yourself giving mostly 4s and 5s, you are being too lenient - RECALIBRATE

IMPORTANT: Return ONLY the JSON object, no additional text before or after."""

    def __init__(self, judge_model, weights: Optional[Dict[str, float]] = None, dataset_name: str = QMSUM_DATASET,
                 criteria_order: Optional[List[int]] = None):
        """Initialize group rubric evaluator.

        Args:
            judge_model: LLM model instance for judging (e.g., GPT, Claude, Gemini)
            weights: Optional weights for each dimension (default: equal weights)
            criteria_order: Optional permutation of range(4) reordering the rubric
                criteria in the judge prompt (judge-prompt perturbation). None keeps
                the original order. Only affects prompt text, not JSON score keys.
        """
        self.judge_model = judge_model
        self.weights = weights or {
            "persona_fidelity": 0.25,
            "originality_specificity": 0.25,
            "relevance_coherence": 0.25,
            "engagement_quality": 0.25
        }
        self.dataset_name = dataset_name
        self.num_retries = 3
        self.criteria_order = criteria_order
        self.RUBRIC = build_rubric_text(GROUP_CRITERIA, criteria_order, GROUP_EPILOGUE)

    @property
    def response_format(self):
        return _make_response_format('group_evaluation', GroupEvalResponse)

    def evaluate(
        self,
        agent_personas: str,
        context_history: str,
        agent_response: str
    ) -> EvaluationResult:
        """Evaluate the entire multi-agent conversation.

        Args:
            agent_personas: Description of all agent personas in the group
            context_history: Conversation history (context + all agent responses)
            agent_response: All agent responses formatted as a single string

        Returns:
            EvaluationResult with scores and reasoning for the entire conversation
        """
        # Build the evaluation prompt
        prompt = self.RUBRIC_TEMPLATE.format(
            agent_personas=agent_personas,
            context_history=context_history,
            agent_response=agent_response,
            RUBRIC=self.RUBRIC,
            RESPONSE_FORMAT=self.RESPONSE_FORMAT
        )

        # Get judgment from the model
        results = self.judge_model.generate_batch(
            prompts=[prompt],
            temperature=0.0,  # Zero temperature for consistent, less lenient evaluations
            max_tokens=500
        )

        # Parse JSON response
        eval_data = self._parse_response(results[0])

        # Calculate weighted reward
        weighted_reward = sum(
            eval_data["scores"].get(dim, 0) * self.weights.get(dim, 0)
            for dim in self.weights.keys()
        )

        # Calculate total reward as average
        total_reward = sum(eval_data["scores"].values()) / len(eval_data["scores"])

        return EvaluationResult(
            scores=eval_data["scores"],
            total_reward=total_reward,
            reasoning=eval_data.get("reasoning", ""),
            weighted_reward=weighted_reward
        )


    def evaluate_batch(
        self,
        evaluation_data: List[Dict[str, str]], agents: dict
    ) -> List[EvaluationResult]:
        """Batch evaluate multiple group conversations in a single LLM call.

        Args:
            evaluation_data: List of dicts, each containing:
                - agent_personas: Description of all agent personas
                - context_history: Conversation history (context + all agent responses)
                - agent_response: All agent responses formatted as a single string

        Returns:
            List of EvaluationResult objects, one per conversation
        """
        prompt, orig, clean = self.build_prompt(evaluation_data, agents)
        if prompt is None:
            return []
        return _generate_and_parse_with_retries(
            self, prompt, evaluation_data, orig, clean, temperature=0.0,
            warn_label="Group rubric evaluator attempt failed",
            fail_message="All retries for group rubric evaluation failed.",
            sleep_seconds=10)

    def build_prompt(self, evaluation_data: Dict, agents: dict) -> Tuple[Optional[str], List[str], List[str]]:
        """Build the evaluation prompt for a single group conversation.

        Returns:
            Tuple of (prompt_string, original_responses, cleaned_responses).
            Returns (None, [], []) if evaluation_data is empty.
        """
        if not evaluation_data:
            return None, [], []

        batch_items = []
        original_responses, cleaned_responses = [], []
        for turn in evaluation_data['turns']:
            original_responses.append(turn['response'])
            if '</think>' in turn['response']:
                response = turn['response'].split('</think>', 1)[-1].strip()
            else:
                response = turn['response']
            cleaned_responses.append(response)
            if turn.get('turn_type', None) == "user":
                batch_items.append(f"[User]: {response[:512]}\n")
            else:
                batch_items.append(f"[{turn['agent_name']} agent]: {response[:1024]}\n")

        batch_content = "\n".join(batch_items)

        prompt = """You are an expert evaluator for multi-agent empathetic dialogue systems.

## Your Task
Score the overall quality of this multi-agent conversation based on how well agents work together as a collective system.

## Evaluation Philosophy
You are evaluating ENTIRE conversations holistically, not individual highlights. Your default assumption is that most conversations are FUNCTIONAL BUT UNREMARKABLE (score 3). Actively identify: repetition, generic therapy-speak, indistinguishable agent voices, imbalanced participation, and predictable patterns.

## Scoring Calibration (STRICT)
• **Score 1-2**: Poor multi-agent coordination
  - Agents sound generic/indistinguishable
  - Contradictory or incoherent conversation flow
  - Dominated by one agent or severe imbalance
  - No meaningful collaborative value

• **Score 3**: DEFAULT - Functional but unremarkable (MOST conversations belong here)
  - Agents maintain basic personas but lack vividness
  - Conventional approaches with some repetition
  - Adequate coherence but predictable flow
  - Each agent contributes something, but nothing memorable

• **Score 4**: Reserved for genuinely strong multi-agent conversations (<20%)
  - Distinct agent voices with minimal overlap
  - Original perspectives with specific insights
  - Good collaborative balance and flow
  - Meaningfully engaging as a collective experience

• **Score 5**: Reserved for exceptional multi-agent conversations (<5%, extremely rare)
  - Each agent perfectly embodies their unique persona throughout
  - Highly novel insights with zero generic language
  - Seamless collaborative orchestration
  - Would impress professional human evaluators

## Evaluation Rubrics
{RUBRIC}

## Specific RED FLAGS (Auto-cap scores at ≤3)
**Persona Fidelity Issues:**
- Agents use identical phrases/structures (e.g., all say "I'm here with you")
- Can't identify which agent said what without labels
- All agents ask similar reflective questions
- The Anchor sounds energetic, The Beacon sounds slow-paced (role reversal)

**Originality Issues:**
- Standard coaching questions appear 3+ times ("What do you think...", "Looking back...")
- Generic therapy language: "holding space," "I understand," "That sounds difficult"
- Agents repeat validation without adding new angles
- Cookie-cutter response templates visible

**Agent Balance Issues:**
- One agent dominates (3+ turns while others have <2)
- Agents respond out of natural order with no justification
- Some agent personas completely missing from conversation
- Imbalanced contribution lengths (one agent 200+ words, others <30 words)

**Coherence Issues:**
- Agents ignore what previous agents said
- Contradictory advice/perspectives offered
- Responses don't build on each other
- Conversation feels like isolated monologues, not dialogue

## Output Format
Return a single JSON object with a "response" array containing exactly ONE evaluation object.
{RESPONSE_FORMAT_BATCH}

## Critical Evaluation Checklist
Before scoring, verify:
1. ✓ Can you identify each agent's unique voice blind (no labels)? NO → Persona ≤3
2. ✓ Count repetitions: Any phrase/structure appears 2+ times? YES → Originality ≤3
3. ✓ Check balance: Does one agent dominate or go missing? YES → Engagement ≤3
4. ✓ Scan for generic language: See 3+ coaching clichés? YES → All scores ≤3
5. ✓ Test coherence: Do agents reference each other's points? NO → Coherence ≤3

## Scoring Decision Tree
- If you see 2+ RED FLAGS → All scores ≤3
- If conversation feels "pretty good" or "adequate" → Score 3 (not 4)
- If debating between two scores → Choose LOWER
- For score 4: Explicitly state what makes this exceptional vs. functional
- For score 5: This should feel like a near-perfect multi-agent orchestration

**Remember**: "Functional," "coherent," and "maintains personas" = baseline expectations (score 3), NOT strengths (score 4).
"""

        if self.dataset_name == QMSUM_DATASET:
            prompt += """
## Multi-Agent Conversation to Evaluate:
{batch_content}

## Human Meeting Context (Use this to assess relevance - are agents addressing the meeting content appropriately?)
{context}
"""
        elif self.dataset_name == EMPATHETIC_DATASET:
            prompt += """
## Human User's Emotional Situation (Use this to assess relevance and empathy - are agents responding appropriately to this context?)
{context}

## Multi-Agent Conversation to Evaluate:
{batch_content}
"""
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        if 'meeting' in evaluation_data:
            context = evaluation_data['meeting']['meeting_context']
        else:
            context = evaluation_data.get('initial_situation', '')

        formatted_prompt = prompt.format(
            RUBRIC=self.RUBRIC,
            RESPONSE_FORMAT_BATCH=self.RESPONSE_FORMAT_BATCH,
            batch_content=batch_content,
            context=context[:1024]
        )

        return formatted_prompt, original_responses, cleaned_responses

    def parse_result(
        self,
        response,
        evaluation_data: Dict,
        original_responses: List[str],
        cleaned_responses: List[str]
    ) -> List[EvaluationResult]:
        """Parse a judge model response into EvaluationResult objects.

        Returns:
            List of EvaluationResult (typically length 1 for group evaluation)
        """
        eval_data = self._parse_response(response)

        if "response" not in eval_data:
            raise ValueError("Response does not contain 'response' array")

        response_list = eval_data["response"]
        assert len(response_list) == 1, \
            f"GroupRubricEvaluator expects exactly 1 result per conversation, but got {len(response_list)}"

        results = []
        for i, resp_eval in enumerate(response_list):
            weighted_reward = sum(
                resp_eval["scores"].get(dim, 0) * self.weights.get(dim, 0)
                for dim in self.weights.keys()
            )
            total_reward = sum(resp_eval["scores"].values()) / len(resp_eval["scores"])

            result = EvaluationResult(
                id=evaluation_data['id'] if self.dataset_name == QMSUM_DATASET else evaluation_data['conv_id'],
                start_idx=evaluation_data['start_idx'] if self.dataset_name == QMSUM_DATASET else None,
                scores=resp_eval["scores"],
                total_reward=total_reward,
                original_response=original_responses[i],
                cleaned_response=cleaned_responses[i],
                reasoning=resp_eval.get("reasoning", ""),
                weighted_reward=weighted_reward
            )
            results.append(result)

        return results

    def _parse_response(self, response) -> Dict[str, Any]:
        """Parse JSON response from judge model (see _parse_judge_response)."""
        return _parse_judge_response(response)
