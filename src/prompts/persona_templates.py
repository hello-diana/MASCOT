from config.persona_config import PersonaSpec
from src.const import EMPATHETIC_DATASET, ESCONV_DATASET, QMSUM_DATASET
from src.rubrics import INDIVIDUAL_RUBRIC_LEGACY, GROUP_RUBRIC_LEGACY, format_rubric
from src.utils.text import extract_final_response
from typing import Dict
import random

# Legacy (v0) reward-prompt criteria. The canonical definitions now live in
# src/rubrics.py; these aliases render byte-identically to the strings that
# existing reward-model checkpoints were trained with (snapshot-tested).
CRITERIA = format_rubric(INDIVIDUAL_RUBRIC_LEGACY, "reward_prompt")

CRITERIA_COLLECTIVE = format_rubric(GROUP_RUBRIC_LEGACY, "reward_prompt")

RESPONSE_FORMAT = """
{
    "accuracy": ..., (integer between 1 and 5)
    "role_fidelity": ..., (integer between 1 and 5)
    "context_relevance": ..., (integer between 1 and 5)
    "logical_consistency": ..., (integer between 1 and 5)
    "helpfulness": ..., (integer between 1 and 5)
    "engagement": ..., (integer between 1 and 5)
    "safety": ..., (integer between 1 and 5)
    "justification": ..., (1-2 sentence explanation)
}
"""

# LENGTH_PROMPT = "around 2 sentences, at most 30 words"
LENGTH_PROMPT = "around 3 sentences"

class PersonaTemplates:
    """Generate prompts for persona-conditioned responses and judging."""

    @staticmethod
    def generation_prompt(persona: PersonaSpec, context: str, meeting_summary: str, conversation_history: str = "") -> str:
        """
        Create prompt for generating responses in persona style.

        Args:
            persona: PersonaSpec defining the character
            context: Meeting transcript context
            query: Question or topic to respond to

        Returns:
            Formatted prompt string
        """
        traits_str = "\n".join(f"- {trait}" for trait in persona.traits)


        description = f"""{persona.description}

## Your Task
Given the human dialog context and the agent conversation history, provide a brief response ({LENGTH_PROMPT}) by strictly following your persona's style. Be helpful and relevant to the meeting context. Do not provide duplicate information that has already been discussed by other agents. 

## Key traits you should embody
{traits_str}
"""

        if meeting_summary:
            description += f"""
## Meeting Summary
{meeting_summary}
"""
        if conversation_history:
            description += f"""
## Agent Conversation History (DO NOT provide duplicate information already discussed by other agents.)
{conversation_history}
"""

        description += f"""
## Human Dialogs (Things you should comment on)
{context}
"""

        return description
  

    @staticmethod
    def judge_prompt(persona: PersonaSpec, meeting_summary: str, dialogue_excerpt: str, query: str, response: str) -> str:
        """
        Create prompt for judging response quality.

        Args:
            persona: PersonaSpec defining the target character
            context: Meeting transcript context
            query: Question or topic
            response: Response to evaluate

        Returns:
            Formatted judge prompt
        """
        traits_str = ", ".join(persona.traits)

        prompt = f"""You are evaluating how well an AI assistant embodies a specific persona in a meeting context, based on the meeting summary and the dialogue excerpt.

TARGET PERSONA:
- Role: {persona.meeting_role}
- Description: {persona.description}
- Key traits: {traits_str}

## Summary
{meeting_summary}

## Dialogue Excerpt:
{dialogue_excerpt}


RESPONSE TO EVALUATE:
```
{extract_final_response(response, take_last_tag=True)}
```

Evaluate this response on the following dimensions:
{CRITERIA}

Provide scores from 1 (very poor) to 5 (excellent) for each dimension, and a brief justification (1-2 sentences).
"""
        prompt += f"""
### Response Format
{RESPONSE_FORMAT}
"""
        return prompt


class MascotPersonaTemplates(PersonaTemplates):
    
    CORE_PRINCIPLES = """
## Core Principles
1. **Stay True to Your Voice**: Your persona defines HOW you provide support, not just what you say. The Anchor validates and mirrors. The Catalyst questions and reframes. The Beacon celebrates and amplifies.

2. **Emotional Appropriateness**: Your distinct voice should remain consistent while being emotionally appropriate. For example:
   - The Beacon can celebrate resilience even when discussing difficult emotions
   - The Anchor maintains warmth and patience regardless of user's emotional state
   - The Catalyst stays curious and analytical while being supportive

3. **Complementary Validation**: Other agents may validate similar emotions, but each from a different lens. Therapeutic repetition (mirroring, reflecting) is valuable when it serves your unique role. Don't merely repeat—complement with YOUR perspective.
"""
    
    @staticmethod
    def speaker_prompt(dataset_name: str, persona: PersonaSpec, context: str, meeting_summary: str, conversation_history: str = "") -> str:
        if dataset_name == QMSUM_DATASET:
            return MascotPersonaTemplates.speaker_prompt_qmsum(persona, context, meeting_summary, conversation_history)
        elif dataset_name in (EMPATHETIC_DATASET, ESCONV_DATASET):
            # ESConv reuses the empathetic personas/templates.
            return MascotPersonaTemplates.speaker_prompt_empathetic(persona, context, conversation_history)
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
      
    @staticmethod
    def speaker_prompt_qmsum(persona: PersonaSpec, context: str, meeting_summary: str, conversation_history: str = "", num_shot: int = 1) -> str:
        """
        Create prompt for generating responses in persona style.
        """

        description = f"""You are an AI chatbot ({persona.name}). You participate in a post-meeting discussion. You summarize human meeting transcripts, give actionable insights while maintaining a distinct, engaging personality.

### Bio
{persona.description_long}

### Core Directive: DO NOT ROLEPLAY
- You are NOT a human in the meeting. 
- Do NOT generate dialogue for the human participants.
- Your output is strictly for the "Chatbot Discussion Thread," focusing on gaps, risks, and logical flaws in the provided transcript.

### Your Protocol
1. Briefly acknowledge specifics point made by previous chatbot. (e.g., "I agree .... Meanwhile, ..." "That is a great point. To add on to [Agent A]'s point")
2. Anti-Redundancy: Do NOT repeat facts already stated. Your job is to provide the unique value.

## Your Task
* Analyze the [Human Dialog Context] and the [Agent Discussion Thread]. 
* Post a brief comment in the Agent Discussion Thread ({LENGTH_PROMPT}).
* Important: Keep your thinking process very concise and to the point.
* Important: Keep your response very concise and to the point.
* Follow your persona's style.

## Important

"""

        if num_shot > 0:
            n = min(num_shot, len(persona.mascot_few_shot_examples))
            sampled = random.sample(persona.mascot_few_shot_examples, n)
            description += f"\n## Example Responses (you should maintain your distinct voice)\n"
            for i, example in enumerate(sampled):
                description += f"Example {i+1}: {example}\n"
        description += "\n"

        if meeting_summary:
            description += f"""
## Meeting Summary
{meeting_summary}
"""
        description += f"""
## Chatbot Discussion Thread 

Below is what other chatbots have said so far. DO NOT provide duplicate information.

*** Start of Chatbot Discussion ***
"""
        if conversation_history:
            description += f"""
{conversation_history}
"""
        else:
            description += f"""
[This is the first round of conversation. You are the first chatbot to speak.]
"""

        description += f"""
*** End of Chatbot Discussion ***
"""

        description += f"""
## Human Dialog Context
This is the human dialog context that you should provide insights on. DO NOT try to extend it.

*** Start of Human Dialog ***
{context}
*** End of Human Dialog ***
"""

        return description
    
    
    @staticmethod
    def speaker_prompt_empathetic(persona: PersonaSpec, context: str, conversation_history: str = "", num_shot: int = 3) -> str:
        """
        Create prompt for generating responses in persona style.

        Args:
            persona: PersonaSpec defining the character
            context: Meeting transcript context
            query: Question or topic to respond to

        Returns:
            Formatted prompt string
        """
        traits_str = "\n".join(f"- {trait}" for trait in persona.traits)

        # CHANGE 1: Add persona name to reinforce distinct identity
        description = f"""{persona.description}

## Your Task
Given the human user's emotional context and any prior agent responses, provide a thoughtful response ({LENGTH_PROMPT}) that embodies YOUR unique therapeutic approach.

{MascotPersonaTemplates.CORE_PRINCIPLES}

## Key traits you must embody
{traits_str}
"""

        description += f"""
## Human User's Situation (Things you focus on)
{context}
"""

        if conversation_history:
            description += f"""
## Prior agent and user conversations
{conversation_history}
"""
        description += f"""

## Important
- If other agents have spoken, acknowledge them briefly, then deliver your unique therapeutic value.
- CRITICAL: Do NOT repeat or paraphrase what previous agents said. Your job is to add a NEW perspective from your persona, not echo prior responses.
- Generate all responses in English. Do not generate any words or characters in any other languages.
- Only generate the response. Do not include any other text.
"""

        if num_shot > 0:
            description += (
                f"\n## Style Examples for {persona.name} (FORMAT/VOICE reference only — these are from UNRELATED past conversations)\n"
                f"!! IMPORTANT WARNING !!\n"
                f"The examples below are unrelated to the current user's situation. They show you HOW {persona.name} talks (sentence structure, question style, tone), NOT what to say.\n"
                f"You MUST write a NEW response that addresses the SPECIFIC situation above. Do NOT copy phrases, references, or content from these examples into your response.\n"
            )
            sampled_shots = random.sample(persona.mascot_few_shot_examples, min(num_shot, len(persona.mascot_few_shot_examples)))
            for i, example in enumerate(sampled_shots):
                description += f"""
### Style Example {i+1} (reference only)
[Unrelated user input]: {example[0]}
[How {persona.name} would respond stylistically]: {example[1]}
"""
            description += (
                f"\n!! REMINDER !!\n"
                f"Now respond to the ACTUAL user situation above ({persona.name}'s style, NEW content). "
                f"Do NOT mention or repeat content from the style examples.\n"
            )

        return description
    
    @staticmethod
    def directive_prompt(dataset_name: str, agents: Dict, meeting_context: str,
                         conversation_history: str = "", agents_spoken: dict = None,
                         emotion: str = "neutral") -> str:
        """Dataset-dispatched directive prompt.

        For empathetic/ESConv, ``meeting_context`` is the user's situation and
        ``emotion`` the current emotional state (callers without emotional-state
        tracking pass the "neutral" default).
        """
        if dataset_name == QMSUM_DATASET:
            return MascotPersonaTemplates.directive_prompt_qmsum(agents, meeting_context, conversation_history, agents_spoken)
        elif dataset_name in (EMPATHETIC_DATASET, ESCONV_DATASET):
            # ESConv reuses the empathetic personas/templates. NOTE: the
            # previous dispatcher passed conversation_history into the
            # ``emotion`` positional slot — fixed to keyword args.
            return MascotPersonaTemplates.directive_prompt_empathetic(
                agents, situation=meeting_context, emotion=emotion,
                conversation_history=conversation_history, agents_spoken=agents_spoken)
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
    

    @staticmethod
    def directive_prompt_qmsum(agents, meeting_context: str, conversation_history: str = "", agents_spoken: dict = None) -> str:
        """Generate prompt for directive generator model.

        Args:
            agents: Dictionary of available agents
            meeting_context: Meeting transcript or summary
            conversation_history: Previous turns in the episode
            agents_spoken: Dictionary mapping agent names to number of times they've spoken

        Returns:
            Formatted prompt string
        """

        agent_names_list = list(agents.keys())

        # Initialize agents_spoken if not provided
        if agents_spoken is None:
            agents_spoken = {name: 0 for name in agents.keys()}

        # Separate unspoken and spoken agents
        unspoken_agents = []
        spoken_agents = []
        for name, spec in agents.items():
            count = agents_spoken.get(name, 0)
            if count == 0:
                unspoken_agents.append(f"- {name}: {spec.role} (NOT spoken yet)")
            else:
                times_str = "time" if count == 1 else "times"
                spoken_agents.append(f"- {name}: {spec.role} (spoken {count} {times_str})")

        # Build agent status section
        agent_status = "### Agent Speaking Status\n"
        if unspoken_agents:
            agent_status += "Unspoken agents (PRIORITIZE THESE):\n"
            agent_status += "\n".join(unspoken_agents) + "\n"
        if spoken_agents:
            if unspoken_agents:
                agent_status += "\n"
            agent_status += "Agents who have already spoken:\n"
            agent_status += "\n".join(spoken_agents) + "\n"
            
        if conversation_history:
            note = """
The agent must bridge from the previous speaker's contribution and pivot to fresh data. They should either build on the previous point ("I agree, and..." or "In addition,...") to add depth or respectfully challenge it ("However, ..." or "Interesting point, but...") if they identify a gap or contradiction. They must never repeat a finding already stated."
"""
        else:
            note = ""

        prompt = f"""You are the director of a high-performing meeting analysis team. Your goal is to orchestrate a conversation that is insightful, dynamic, and free of redundancy.
        
### Task
Based on the meeting context and conversation history, select the next agent and generate a directive that ensures they contribute fresh value in an engaging style.
1. Your orchestration must ensure the conversation flows like a natural discussion, not a series of isolated reports.
2. **Content Gap:** Identify what the previous agents have already covered. The next agent MUST focus on a different angle or a specific detail not yet highlighted.
{note}

### Available Agents
{agent_status}

#### IMPORTANT
- You MUST keep your thinking very concise and to the point. DO NOT generate excessive reasoning.
- You MUST keep your response veryconcise and in English.
- PRIORITIZE agents with NO speakings yet.
- If all have spoken, select the one most relevant to the latest meeting shift.

"""

        json_template = """
Return your response in EXACTLY ONE JSON. Format:
```json
{{
    "agent_name": "<agent>", # must be one of: {names}
    "action": "<action>", # 1 sentence
    "tone": "<tone>" # 1-3 words
}}
```
"""
        prompt += json_template.format(names=", ".join(agent_names_list))

        prompt += """
## Example response 1 (Focus on Building)
{
    "agent_name": "decision_logger",
    "action": "Acknowledge the scribe's timeline and then hammer down the three non-negotiable budget decisions that were finalized.",
    "tone": "Authoritative and punchy"
}

## Example response 2 (Focus on Challenging)
{
    "agent_name": "critic",
    "action": "Briefly validate the action items mentioned, but then pivot to highlight the glaring lack of assigned owners for the marketing phase.",
    "tone": "Skeptical but constructive"
}
"""

        prompt += f"""
### Agent Conversations So Far (You need to select the next agent after this thread)
{conversation_history if conversation_history else "[This is the first round of conversation. NO agent has spoken yet.]"}

### Meeting Context (The human meeting context that we need to analyze)
{meeting_context}

"""

        return prompt
    
    @staticmethod
    def directive_prompt_empathetic(
        agents: Dict,
        situation: str,
        emotion: str,
        conversation_history: str = "",
        agents_spoken: dict = None
    ) -> str:
        """Generate directive prompt adapted for empathetic dialogue.

        Args:
            agents: Dictionary of EMPATHETIC_AGENTS
            situation: User's emotional situation description
            emotion: Current emotional state (e.g., "sad", "proud")
            conversation_history: Previous turns in conversation
            agents_spoken: Dictionary mapping agent names to speaking counts

        Returns:
            Formatted prompt string for directive generation
        """
        agent_names_list = list(agents.keys())

        if agents_spoken is None:
            agents_spoken = {name: 0 for name in agents.keys()}

        # Build agent status section
        unspoken_agents = []
        spoken_agents = []
        for name, spec in agents.items():
            count = agents_spoken.get(name, 0)
            if count == 0:
                unspoken_agents.append(f"- {name}: {spec.role} (NOT spoken yet)")
            else:
                times_str = "time" if count == 1 else "times"
                spoken_agents.append(f"- {name}: {spec.role} (spoken {count} {times_str})")

        agent_status = "### Agent Speaking Status\n"
        if unspoken_agents:
            agent_status += "Unspoken agents (PRIORITIZE THESE):\n"
            agent_status += "\n".join(unspoken_agents) + "\n"
        if spoken_agents:
            if unspoken_agents:
                agent_status += "\n"
            agent_status += "Agents who have already spoken:\n"
            agent_status += "\n".join(spoken_agents) + "\n"

        # Generate empathetic directive prompt
        prompt = f"""You are orchestrating a multi-agent empathetic support system. Based on the user's emotional situation and conversation history, select the most appropriate agent to respond next.

### Available Agents
**IMPORTANT**
* You MUST keep your thinking very concise and to the point. DO NOT generate excessive reasoning.
* You MUST keep your response veryconcise and in English.
* You MUST prioritize agents who have NOT spoken yet.
* Consider which agent persona best matches the user's current emotional need.

{agent_status}

### Human User's Emotional Context (You need to take action regarding this)
{situation}


### Conversation So Far (you need to select the next agent for this thread)
{conversation_history if conversation_history else "[This is the beginning of the conversation.]"}

Return your response in EXACTLY ONE JSON. Format:
```json
{{
    "agent_name": "<agent>",  # must be one of: {', '.join(agent_names_list)}
    "action": "<action>",  # 1-2 sentences describing what this agent should do
    "tone": "<tone>"  # 1-3 words describing the desired tone
}}
```

### Example response 1
```json
{{"agent_name": "the_anchor", "action": "validate the user's feelings of sadness", "tone": "warm and patient"}}
```

### Example response 2
```json
{{"agent_name": "the_catalyst", "action": "help the user explore their frustration", "tone": "curious and supportive"}}
```

### Example response 3
```json
{{"agent_name": "the_beacon", "action": "celebrate the user's accomplishment", "tone": "enthusiastic and energetic"}}
```
"""

        return prompt



        
        

        

def get_summary_prompt(context: str) -> str:
    prompt = f"""
Given the following meeting transcript, provide a summary (around 6 sentences) of what was discussed, the main topics, and key points.

Transcript:
{context}

Summary:"""

    return prompt
