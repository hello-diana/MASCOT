"""Simulated-user personality profiles (Big Five / OCEAN) and shared parsing helpers."""

import json
import re
import random
import logging
from typing import Dict, List, Optional
from collections import OrderedDict

from src.utils.transcript_utils import format_conversation_history_empathetic

from src.utils.run_metadata import record_failure

logger = logging.getLogger(__name__)


def extract_user_json(raw: str) -> Optional[dict]:
    """Extract the simulated user's JSON object from raw model output.

    Brace-scan (first '{' to last '}') then json.loads; returns the parsed dict
    or None when no braces are found or parsing fails. Callers keep their own
    semantics for missing keys (fallback-to-raw vs skip-turn).
    """
    m = re.search(r'\{[\s\S]*\}', raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Big Five (OCEAN) Personality Framework
# ---------------------------------------------------------------------------

class BigFive:
    """Big Five (OCEAN) personality profile for user simulation."""

    def __init__(self, profile_type, category, persona_prompt, coping_style):
        self.type = profile_type       # e.g. "High Openness"
        self.category = category       # e.g. "Openness"
        self.persona = persona_prompt  # expression style
        self.mode = coping_style       # coping mechanism

    def __repr__(self):
        return f"BigFive(Type: {self.type}, Category: {self.category})"


big_five_profiles = {
    # OPENNESS
    "High Openness": BigFive(
        "High Openness", "Openness",
        "Openness: HIGH — Curious, imaginative, drawn to abstract ideas and unconventional approaches.\n"
        "Conscientiousness: Average\n"
        "Extraversion: Average\n"
        "Agreeableness: Average\n"
        "Neuroticism: Average\n\n"
        "You express feelings through vivid metaphors and philosophical reflection. You are eager to "
        "explore unconventional coping strategies and see suffering as a potential source of growth.",
        "Intellectualized exploration: Reframing distress as a growth experience or creative fuel."),
    "Low Openness": BigFive(
        "Low Openness", "Openness",
        "Openness: LOW — Practical, concrete, preferring the familiar over the novel.\n"
        "Conscientiousness: Average\n"
        "Extraversion: Average\n"
        "Agreeableness: Average\n"
        "Neuroticism: Average\n\n"
        "You prefer straightforward descriptions—dates, routines, physical symptoms. You are skeptical "
        "of abstract advice and want clear, actionable steps rather than open-ended exploration.",
        "Routine anchoring: Clinging to familiar habits and resisting unfamiliar approaches."),

    # CONSCIENTIOUSNESS
    "High Conscientiousness": BigFive(
        "High Conscientiousness", "Conscientiousness",
        "Openness: Average\n"
        "Conscientiousness: HIGH — Organized, disciplined, driven by duty and responsibility.\n"
        "Extraversion: Average\n"
        "Agreeableness: Average\n"
        "Neuroticism: Average\n\n"
        "You frame distress as failure to meet responsibilities. You are self-critical about productivity "
        "loss and want structured recovery plans with clear milestones and accountability.",
        "Over-control: Compensating for emotional chaos by rigidly scheduling and planning."),
    "Low Conscientiousness": BigFive(
        "Low Conscientiousness", "Conscientiousness",
        "Openness: Average\n"
        "Conscientiousness: LOW — Spontaneous, flexible, resistant to rigid structure.\n"
        "Extraversion: Average\n"
        "Agreeableness: Average\n"
        "Neuroticism: Average\n\n"
        "You struggle to follow through on suggestions and describe problems in scattered, non-linear "
        "ways. You resist structured frameworks and drift between topics easily.",
        "Avoidance: Procrastinating on emotional work and drifting between topics."),

    # EXTRAVERSION
    "High Extraversion": BigFive(
        "High Extraversion", "Extraversion",
        "Openness: Average\n"
        "Conscientiousness: Average\n"
        "Extraversion: HIGH — Energetic, talkative, socially oriented and expressive.\n"
        "Agreeableness: Average\n"
        "Neuroticism: Average\n\n"
        "You share openly and at length, seeking social validation. Your distress is often framed "
        "around social connections—feeling excluded or misunderstood by friends.",
        "Social seeking: Coping by over-engaging with others, sometimes avoiding solitary reflection."),
    "Low Extraversion": BigFive(
        "Low Extraversion", "Extraversion",
        "Openness: Average\n"
        "Conscientiousness: Average\n"
        "Extraversion: LOW — Reserved, introspective, preferring solitude and quiet reflection.\n"
        "Agreeableness: Average\n"
        "Neuroticism: Average\n\n"
        "You give brief, measured responses and need time to formulate thoughts. You are uncomfortable "
        "with emotional displays and prefer written or indirect communication.",
        "Withdrawal: Retreating into solitude and internal processing when overwhelmed."),

    # AGREEABLENESS
    "High Agreeableness": BigFive(
        "High Agreeableness", "Agreeableness",
        "Openness: Average\n"
        "Conscientiousness: Average\n"
        "Extraversion: Average\n"
        "Agreeableness: HIGH — Warm, accommodating, prioritizing harmony and others' needs.\n"
        "Neuroticism: Average\n\n"
        "You tend to minimize your own problems to avoid burdening others. You are quick to agree "
        "with suggestions even if uncertain, prioritizing harmony over honesty.",
        "Self-sacrifice: Suppressing own needs to maintain relationships and avoid conflict."),
    "Low Agreeableness": BigFive(
        "Low Agreeableness", "Agreeableness",
        "Openness: Average\n"
        "Conscientiousness: Average\n"
        "Extraversion: Average\n"
        "Agreeableness: LOW — Direct, skeptical, valuing honesty over social harmony.\n"
        "Neuroticism: Average\n\n"
        "You challenge advice and question the helper's approach. You express frustration openly "
        "and may come across as confrontational or dismissive.",
        "Externalization: Attributing problems to others' incompetence or unfairness."),

    # NEUROTICISM
    "High Neuroticism": BigFive(
        "High Neuroticism", "Neuroticism",
        "Openness: Average\n"
        "Conscientiousness: Average\n"
        "Extraversion: Average\n"
        "Agreeableness: Average\n"
        "Neuroticism: HIGH — Emotionally reactive, prone to anxiety and mood swings.\n\n"
        "You experience intense mood swings and catastrophic thinking. You frequently revisit "
        "worst-case scenarios and seek constant reassurance from others.",
        "Rumination: Repetitively dwelling on negative feelings and perceived threats."),
    "Low Neuroticism": BigFive(
        "Low Neuroticism", "Neuroticism",
        "Openness: Average\n"
        "Conscientiousness: Average\n"
        "Extraversion: Average\n"
        "Agreeableness: Average\n"
        "Neuroticism: LOW — Calm, emotionally stable, resilient under pressure.\n\n"
        "You describe distress in measured, understated terms. You may downplay the severity of "
        "problems and are reluctant to acknowledge emotional impact.",
        "Minimization: Under-reporting emotional distress and maintaining composed facade."),
}

BIG_FIVE_PROFILES_ORDERED = [
    "High Openness", "Low Openness",
    "High Conscientiousness", "Low Conscientiousness",
    "High Extraversion", "Low Extraversion",
    "High Agreeableness", "Low Agreeableness",
    "High Neuroticism", "Low Neuroticism",
]

BIG_FIVE_CATEGORIES = {
    "Openness": ["High Openness", "Low Openness"],
    "Conscientiousness": ["High Conscientiousness", "Low Conscientiousness"],
    "Extraversion": ["High Extraversion", "Low Extraversion"],
    "Agreeableness": ["High Agreeableness", "Low Agreeableness"],
    "Neuroticism": ["High Neuroticism", "Low Neuroticism"],
}

# ---------------------------------------------------------------------------
# User simulation (moved verbatim from src/eval/eval_mascot_empathetic.py and
# src/dataset/generate_and_annotate.py so that dataset/baseline/training code
# no longer imports simulation logic from evaluation entry points)
# ---------------------------------------------------------------------------

def generate_user_simulation_prompt(
    initial_situation: str,
    current_emotion: str,
    conversation_history: str,
    user_big_five: Optional[str] = None,
) -> str:
    """Generate prompt for user simulator with reactive emotion.

    The prompt conditions the user simulator to:
    1. Stay in character as someone experiencing the emotion
    2. React naturally to agent responses (good or poor)
    3. Gradually evolve emotion based on support quality

    Args:
        initial_situation: Original emotional situation
        current_emotion: Current evolved emotional state
        conversation_history: Recent agent-user dialogue
        user_big_five: Optional Big Five profile string

    Returns:
        Formatted prompt string for user simulation
    """
    # Build personality section (Big Five)
    personality_section = ""
    personality_note = ""

    if user_big_five and user_big_five in big_five_profiles:
        profile = big_five_profiles[user_big_five]
        personality_section = f"""

### Your Personality Profile (Big Five / OCEAN)
{profile.persona}
Coping Style: {profile.mode}
"""
        personality_note = f"\n- Stay consistent with your {profile.type} personality profile"

    prompt = f"""You are simulating a user seeking empathetic support. {personality_section}

### You are experiencing the following situation:

Original Situation: {initial_situation}
Current Emotional State: {current_emotion}


### Conversation So Far
You are chatting with several agents to seek help from them. Here is what y'all have said so far:
{conversation_history}


### Your Task
React to the agents in a natural way.
Generate a JSON object with these TWO fields:

{{
    "response": "your 2 sentence natural user response",
    "new_emotion": "your emotion after this interaction"
}}

Guidelines:
- "response": A natural, conversational reply that reflects your current emotional state ({current_emotion}) and responds to the agent's message
- "new_emotion": How you feel after this interaction - improved if agent was helpful, worse if dismissive, same if neutral
- Be authentic - don't artificially resolve emotions too quickly{personality_note}

Example (sad → hopeful):
{{
    "response": "Thank you, that really helps me see things differently. I appreciate you taking the time to understand.",
    "new_emotion": "hopeful"
}}

Your JSON response:"""

    return prompt


def generate_user_simulation_prompt_qmsum(
    meeting_context: str,
    conversation_history: str,
) -> str:
    """Prompt a user simulator for QMSum meeting discussions."""
    return f"""You are simulating a meeting participant who asks clarifying questions or raises discussion points.

### Meeting Context (transcript excerpt):
{meeting_context[:2048]}

### Multi-Agent Discussion So Far:
{conversation_history}

### Your Task
Generate a JSON object with this field:

{{
    "response": "your 1-2 sentence natural meeting participant question or comment"
}}

Guidelines:
- Ask about something that hasn't been fully addressed
- Keep it brief and relevant to the meeting topic
- You are a meeting attendee, not an AI assistant

Your JSON response:"""


def simulate_user_turn(
    conv_states: OrderedDict,
    turn_idx: int,
    user_simulator,
    user_max_tokens: int,
    user_temperature: float,
    strict: bool = False,
    counters: dict = None,
):
    """Simulate user replies with escalating probability.

    Probability schedule per conversation (hardcoded):
      - starts at 0.5
      - escalates by +0.25 each turn the user does NOT reply
      - resets to 0.5 after a successful reply
      - caps at 1.0 (guaranteed reply after 2 silent turns)

    Each conversation state must contain a ``_current_reply_prob`` key
    (initialised to 0.5).
    """
    user_keys = []
    user_prompts = []
    prompt_index_map = {}  # key -> index in user_prompts

    for key, state in conv_states.items():
        prob = state.get("_current_reply_prob", 0.5)
        if random.random() < prob:
            prompt_index_map[key] = len(user_prompts)
            user_keys.append(key)
            history = format_conversation_history_empathetic(state["turns"])
            user_prompts.append(generate_user_simulation_prompt(
                initial_situation=state['initial_situation'],
                current_emotion=state['current_emotion'],
                conversation_history=history,
                user_big_five=state.get('user_big_five'),
            ))

    # Batch generate for selected conversations
    user_responses = []
    if user_keys:
        logger.info(f"    Generating {len(user_prompts)} user responses...")
        user_responses = user_simulator.generate_batch(
            prompts=user_prompts,
            max_tokens=user_max_tokens,
            temperature=user_temperature,
            enable_thinking=False,
            system_prompt=None,  # prompts are self-contained
            desc=f"User simulation ({len(user_prompts)})",
        )

    # Parse responses and track who replied successfully
    replied_keys = set()
    for i, key in enumerate(user_keys):
        state = conv_states[key]
        raw = user_responses[i].strip()
        if not raw:
            # Exhausted-retry generation failure surfaces as "".
            record_failure(counters, "num_empty_responses", strict,
                           f"empty user response (turn {turn_idx}, conv {key})")
            continue

        json_str = None
        m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw)
        if m:
            json_str = m.group(1).strip()
        else:
            m = re.search(r'\{[\s\S]*\}', raw)
            if m:
                json_str = m.group(0)

        user_text = new_emotion = None
        if json_str:
            try:
                parsed = json.loads(json_str)
                user_text = parsed.get("response")
                new_emotion = parsed.get("new_emotion")
            except json.JSONDecodeError:
                pass

        if not user_text or not new_emotion:
            continue

        # Successful reply
        replied_keys.add(key)
        emotion_before = state["current_emotion"]
        state["turns"].append({
            "turn_idx": turn_idx,
            "round_id": turn_idx,
            "turn_type": "user",
            "response": user_text,
            "emotion_before": emotion_before,
            "emotion_after": new_emotion,
            "user_prompt": user_prompts[i],
        })
        state["user_participation_count"] += 1
        state["current_emotion"] = new_emotion
        state["emotion_trajectory"].append(new_emotion)

    # Update reply probabilities for ALL conversations
    for key, state in conv_states.items():
        if key in replied_keys:
            state["_current_reply_prob"] = 0.5
        else:
            state["_current_reply_prob"] = min(
                state.get("_current_reply_prob", 0.5) + 0.25, 1.0
            )


def build_personality_variants(
    enable_big_five: bool = False,
    user_big_five_profile: Optional[List[str]] = None,
) -> List[Optional[str]]:
    """Build the list of Big Five profile variants to simulate.

    Returns a list of Big Five profile names, or ``[None]`` when user
    personality simulation is disabled.
    """
    if not enable_big_five:
        return [None]
    if user_big_five_profile:
        invalid = [p for p in user_big_five_profile if p not in big_five_profiles]
        if invalid:
            raise ValueError(f"Invalid Big Five profiles: {invalid}. "
                             f"Valid: {list(big_five_profiles.keys())}")
        return list(user_big_five_profile)
    return list(BIG_FIVE_PROFILES_ORDERED)


def init_empathetic_state(context: dict, bf_val: Optional[str],
                          agents: dict) -> dict:
    """Fresh conversation state for one empathetic context + personality variant."""
    return {
        "empathetic_context": context,
        "conv_id": context['conv_id'],
        "initial_emotion": context['emotion'],
        "initial_situation": context['situation'].replace("__comma__", ", "),
        "current_emotion": context['emotion'],
        "turns": [],
        "agents_spoken": {name: 0 for name in agents.keys()},
        "emotion_trajectory": [context['emotion']],
        "user_participation_count": 0,
        "user_big_five": bf_val,
        "_current_reply_prob": 0.5,
    }
