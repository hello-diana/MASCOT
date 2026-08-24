
from src.rubrics import PAPER_V1, format_rubric, get_rubric
from src.utils.data_utils import clean_response


def get_prompt_persona_reward_model(persona: str, persona_spec, response: str,
                                    rubric_version: str = PAPER_V1) -> str:
    """Persona reward-model scoring prompt.

    ``rubric_version`` selects the criteria text. New training uses
    ``paper_v1``; scoring with a checkpoint trained on the legacy criteria
    requires ``legacy_v0`` (resolved from checkpoint metadata or an explicit
    flag — see ``src.rubrics.resolve_checkpoint_rubric_version``).
    """
    criteria = format_rubric(get_rubric("individual", rubric_version), "reward_prompt")
    prompt = f"""Please evaluate the following response (e.g. how well the model generates the answers and follows its role description) based on the role of the persona:

Persona: {persona}
Role: {persona_spec.role}

## Criteria
{criteria}

## Response
```
{response}
```
"""
    return prompt


def get_prompt_collective_reward_model(turns: list[dict],
                                       rubric_version: str = PAPER_V1) -> str:
    """Group reward-model scoring prompt (see rubric_version note above)."""
    criteria = format_rubric(get_rubric("group", rubric_version), "reward_prompt")
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
{criteria}
"""

    return prompt
