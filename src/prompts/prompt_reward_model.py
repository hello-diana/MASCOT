
from src.prompts.persona_templates import CRITERIA, CRITERIA_COLLECTIVE
from src.utils.data_utils import clean_response


def get_prompt_persona_reward_model(persona: str, persona_spec, response: str) -> str:
    prompt = f"""Please evaluate the following response (e.g. how well the model generates the answers and follows its role description) based on the role of the persona:

Persona: {persona}
Role: {persona_spec.role}

## Criteria
{CRITERIA}

## Response
```
{response}
```
"""
    return prompt


def get_prompt_collective_reward_model(turns: list[dict]) -> str:
    conversation_history = ""
    for turn in turns:
        if turn.get('directive') is not None:
            d = turn['directive']
            action = d.get('action', d.get('action_type', None))
            
            if action is None:
                action = "Respond in the thread."
                
            conversation_history += (
                f"(directive ({d['agent_name']}): {action}. "
                f"Tone: {d.get('tone', 'neutral')})\n"
            )
        agent_name = turn.get('agent_name', 'Unknown')
        response = turn.get('response', turn.get('agent_response', ''))
        cleaned = clean_response(response)
        conversation_history += f"{agent_name}: {cleaned}\n"

    prompt = f"""You are an expert evaluator for a multi-agent social companion system. Your task is to score the overall quality of a multi-agent conversation.

You should focus on both the quality of the directives and the collective conversations of the agents.

## Multi-Agent Conversation
{conversation_history}

## Criteria
{CRITERIA_COLLECTIVE}
"""

    return prompt
