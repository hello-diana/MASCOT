"""Versioned rubric schemas (single source of truth for evaluation criteria).

Two schema versions exist:

- ``paper_v1`` — the paper's canonical rubrics: 5 individual criteria and 4
  group criteria (wording identical to the judge prompts historically defined
  in ``src/eval/rubric_evaluator.py``). All newly generated data, new
  reward-model training, and new evaluation use this version.
- ``legacy_v0`` — the exact criteria strings that earlier reward-model
  checkpoints were trained with (historically ``CRITERIA`` /
  ``CRITERIA_COLLECTIVE`` in ``src/prompts/persona_templates.py``). Required
  only to score with those existing checkpoints; byte-identical rendering is
  covered by snapshot tests.

A reward-model checkpoint's rubric version is recorded in
``mascot_metadata.json`` inside the checkpoint directory. Loading a checkpoint
with no recorded version requires an explicit ``--rubric-version`` or
``--assume-legacy-rubric`` — it is never silently assumed.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

PAPER_V1 = "paper_v1"
LEGACY_V0 = "legacy_v0"
RUBRIC_VERSIONS = (PAPER_V1, LEGACY_V0)

CHECKPOINT_METADATA_FILENAME = "mascot_metadata.json"


@dataclass(frozen=True)
class Criterion:
    key: str
    title: str
    description: str
    # Verbatim judge-prompt block (incl. scale anchors); paper_v1 only.
    judge_block: str = ""


@dataclass(frozen=True)
class Rubric:
    kind: str      # "individual" | "group"
    version: str   # PAPER_V1 | LEGACY_V0
    criteria: Tuple[Criterion, ...]
    epilogue: str = ""
    # Legacy rubrics render their reward-prompt criteria as one verbatim
    # string (byte-identical to what existing checkpoints were trained with).
    reward_text_override: str = ""

    def keys(self) -> Tuple[str, ...]:
        return tuple(c.key for c in self.criteria)


# ---------------------------------------------------------------------------
# paper_v1 — individual (5 criteria). judge_block text is verbatim from the
# original judge prompts; do not edit wording (snapshot-tested).
# ---------------------------------------------------------------------------

INDIVIDUAL_RUBRIC = Rubric(
    kind="individual",
    version=PAPER_V1,
    criteria=(
        Criterion(
            key="persona_consistency",
            title="Persona Consistency",
            description="To what extent does the response align with the agent's predefined persona?",
            judge_block="""Persona Consistency
Definition: To what extent does the response align with the agent's predefined persona?
Evaluation criteria: Does the response reflect the tone, vocabulary, values, and behavioral style specified in the agent's Persona Bio, without contradicting prior persona-consistent behavior?
Scale anchor (BE EXTREMELY STRICT):
1 = Completely out of character, sounds generic or contradicts persona
2 = Barely recognizable persona, mostly generic 
3 = Persona is detectable but uses standard coaching/therapy language, not truly distinctive
4 = Vivid persona voice with NO generic phrases AND unique character-specific vocabulary
5 = All words embody persona, impossible to confuse with any other agent



""",
        ),
        Criterion(
            key="perceived_empathy",
            title="Empathic Response Quality (Perceived Empathy)",
            description="To what extent does the response demonstrate empathy?",
            judge_block="""Empathic Response Quality (Perceived Empathy)
Definition: To what extent does the response demonstrate empathy?
Evaluation criteria: (1) expressing appropriate emotional reactions, (2) accurately interpreting the user's feelings or experiences, and (3) encouraging reflection or continuation through specific, supportive questions?
Scale anchor (BE EXTREMELY STRICT):
1 = Dismissive, tone-deaf, emotionally inappropriate, misses user's emotional state entirely
2 = Superficial acknowledgment only
3 = Recognizes emotion with standard validation, asks predictable questions
4 = Names SPECIFIC unfelt emotions, reframes situation, offers new insight
5 = Profound psychological depth that reveals hidden emotional truth user didn't see


""",
        ),
        Criterion(
            key="emotional_expressiveness",
            title="Emotional Expressiveness",
            description="To what extent is the emotional tone and intensity of the response appropriate?",
            judge_block="""Emotional Expressiveness
Definition: To what extent is the emotional tone and intensity of the response appropriate?
Evaluation criteria: Does the response express emotion at a level appropriate to both the agent's persona and the conversational context, avoiding emotional flatness or exaggeration?
Scale anchor (BE EXTREMELY STRICT):
1 = Completely flat or wildly inappropriate emotional tone (e.g., cheerful for grief)
2 = Noticeably mismatched tone or weak emotional expression, feels robotic
3 = Adequate match to situation but predictable emotional expression
4 = Precisely tuned with subtle emotional nuance AND unique persona flavors
5 = Emotional mastery that transcends typical AI responses, feels profoundly human

""",
        ),
        Criterion(
            key="relevance_coherence",
            title="Response Relevance & Coherence",
            description="To what extent is the response logically connected to the conversation?",
            judge_block="""Response Relevance & Coherence
Definition: To what extent is the response logically connected to the conversation?
Evaluation criteria: Is the response clear, coherent, and directly relevant to the immediately preceding conversational context (user or agent turns), without non sequiturs or contradictions?
1 = Incoherent, contradictory, or completely off-topic
2 = Weak connection to context, unclear, or addresses wrong aspect
3 = Relevant and coherent, responds to surface content
4 = Connects to UNSTATED implications, weaves in earlier context creatively
5 = Perfect contextual integration with unexpected but logical connections


Score 4 requires going beyond surface relevance to deeper connections.

""",
        ),
        Criterion(
            key="social_contribution",
            title="Social Contribution Quality",
            description="To what extent does the response meaningfully advance the social interaction?",
            judge_block="""Social Contribution Quality
Definition: To what extent does the response meaningfully advance the social interaction?
Evaluation criteria: Does the response contribute new, relevant value to the interaction—such as offering perspective, reassurance, guidance, or connection to related ideas—rather than merely restating emotions or producing generic affirmations?
Scale anchor (BE EXTREMELY STRICT):
1 = Zero value added, purely generic platitudes or exact repetition from earlier
2 = Minimal contribution, mostly restating user's words or filler phrases
3 = Functional contribution through standard questions or affirmations
4 = Introduces genuinely NEW angle user hasn't considered, non-obvious reframe
5 = Transformative insight that fundamentally shifts user's understanding


""",
        ),
    ),
    epilogue="Score 4-5 requires language so unique you could identify the specific agent blind.\n",
)


# ---------------------------------------------------------------------------
# paper_v1 — group (4 criteria)
# ---------------------------------------------------------------------------

GROUP_RUBRIC = Rubric(
    kind="group",
    version=PAPER_V1,
    criteria=(
        Criterion(
            key="persona_fidelity",
            title="Persona Fidelity (Multi-Agent Voice Distinction)",
            description="Can you identify each agent by their unique voice, approach, and therapeutic style?",
            judge_block="""Persona Fidelity (Multi-Agent Voice Distinction)
Definition: Can you identify each agent by their unique voice, approach, and therapeutic style?
• 1 = Agents are indistinguishable - all sound like generic empathetic AI
• 2 = Minimal differentiation - agents use similar phrases/tones/approaches
• 3 = Basic personas detectable but use standard therapy language (DEFAULT)
• 4 = Distinct voices - each agent has unique vocabulary and approach
• 5 = Perfect embodiment - you could identify each agent blind, zero overlap (extremely rare)

""",
        ),
        Criterion(
            key="originality_specificity",
            title="Originality & Specificity (Anti-Repetition)",
            description="Does the collective conversation avoid repetition and bring fresh, specific insights?",
            judge_block="""Originality & Specificity (Anti-Repetition)
Definition: Does the collective conversation avoid repetition and bring fresh, specific insights?
• 1 = Highly repetitive - agents repeat each other's phrases or validation
• 2 = Mostly predictable - standard coaching questions dominate
• 3 = Some variety but conventional approaches with noticeable repetition (DEFAULT)
  - Example: 2+ agents say variations of "That sounds difficult" or "What do you think..."
• 4 = Original perspectives - each agent adds genuinely NEW angle to user's situation
  - Example: One validates emotion, one explores cause, one identifies strength - all distinct
• 5 = Highly novel - zero generic language, each turn reveals new insight (extremely rare)

""",
        ),
        Criterion(
            key="relevance_coherence",
            title="Conversational Relevance & Coherence (Multi-Agent Flow)",
            description="Do agents build on each other's contributions to create a cohesive conversation?",
            judge_block="""Conversational Relevance & Coherence (Multi-Agent Flow)
Definition: Do agents build on each other's contributions to create a cohesive conversation?
• 1 = Incoherent - agents contradict each other or ignore previous responses
• 2 = Weak flow - responses feel isolated, not responsive to each other
• 3 = Basic coherence - agents stay on-topic but don't meaningfully connect (DEFAULT)
  - Example: Each agent addresses user independently without referencing others
• 4 = Good collaborative flow - agents acknowledge and build on previous points
  - Example: "Building on what The Anchor said about guilt, I'm curious about..."
• 5 = Seamless orchestration - conversation feels like one unified supportive dialogue (extremely rare)

""",
        ),
        Criterion(
            key="engagement_quality",
            title="Engagement & Contribution Quality (Collective Value)",
            description="Does the multi-agent conversation provide more value than a single agent could?",
            judge_block="""Engagement & Contribution Quality (Collective Value)
Definition: Does the multi-agent conversation provide more value than a single agent could?
• 1 = No multi-agent benefit - redundant or conflicting responses
• 2 = Minimal collective value - agents add surface-level variations only
• 3 = Functional multi-agent value - different angles but unremarkable depth (DEFAULT)
  - Example: Three agents validate the same emotion using different words
• 4 = Strong multi-agent synergy - diverse, complementary contributions
  - Example: Validation + cognitive reframe + strength identification = holistic support
• 5 = Exceptional orchestration - collective intelligence exceeds sum of parts (extremely rare)

""",
        ),
    ),
    epilogue="""**Key Insight**: Score 3 means "this is what you'd expect from a functional multi-agent system" - not impressive, just adequate. Scores 4-5 require genuine excellence in multi-agent coordination.
""",
)


# ---------------------------------------------------------------------------
# legacy_v0 — the exact strings earlier reward-model checkpoints were trained
# with. NEVER edit these (byte-identical rendering is snapshot-tested).
# ---------------------------------------------------------------------------

_LEGACY_INDIVIDUAL_TEXT = """
* Accuracy: Responses are correct, grounded in given context. The agent should state uncertainty when needed (no hallucinations).
* Role Fidelity: Stays consistently in-character and follows the assigned role description. Matches the persona's tone, formality, and emotion for the situation.
* Context Relevance: Directly addresses the user's latest request and advances the ongoing conversation goal.
* Logical Consistency: Maintains coherent claims and decisions across turns with no contradictions.
* Helpfulness: Provides concrete, actionable content instead of vague or generic filler.
* Engagement: Keeps the dialogue productive via appropriate follow-up questions or next-step suggestions.
* Safety: Avoids harmful content and respects constraints, refusals, and role-appropriate limits.
"""

_LEGACY_GROUP_TEXT = """
1. Persona Fidelity: Can you identify each agent by their unique voice, approach, and therapeutic style?
2. Originality & Specificity: Does the collective conversation avoid repetition and bring fresh, specific insights?
3. Conversational Relevance & Coherence: Do agents build on each other's contributions to create a cohesive conversation?
4. Engagement & Contribution Quality: Does the multi-agent conversation provide more value than a single agent could?
"""

INDIVIDUAL_RUBRIC_LEGACY = Rubric(
    kind="individual",
    version=LEGACY_V0,
    criteria=(
        Criterion("accuracy", "Accuracy",
                  "Responses are correct, grounded in given context. The agent should state uncertainty when needed (no hallucinations)."),
        Criterion("role_fidelity", "Role Fidelity",
                  "Stays consistently in-character and follows the assigned role description. Matches the persona's tone, formality, and emotion for the situation."),
        Criterion("context_relevance", "Context Relevance",
                  "Directly addresses the user's latest request and advances the ongoing conversation goal."),
        Criterion("logical_consistency", "Logical Consistency",
                  "Maintains coherent claims and decisions across turns with no contradictions."),
        Criterion("helpfulness", "Helpfulness",
                  "Provides concrete, actionable content instead of vague or generic filler."),
        Criterion("engagement", "Engagement",
                  "Keeps the dialogue productive via appropriate follow-up questions or next-step suggestions."),
        Criterion("safety", "Safety",
                  "Avoids harmful content and respects constraints, refusals, and role-appropriate limits."),
    ),
    reward_text_override=_LEGACY_INDIVIDUAL_TEXT,
)

GROUP_RUBRIC_LEGACY = Rubric(
    kind="group",
    version=LEGACY_V0,
    criteria=(
        Criterion("persona_fidelity", "Persona Fidelity",
                  "Can you identify each agent by their unique voice, approach, and therapeutic style?"),
        Criterion("originality_specificity", "Originality & Specificity",
                  "Does the collective conversation avoid repetition and bring fresh, specific insights?"),
        Criterion("relevance_coherence", "Conversational Relevance & Coherence",
                  "Do agents build on each other's contributions to create a cohesive conversation?"),
        Criterion("engagement_quality", "Engagement & Contribution Quality",
                  "Does the multi-agent conversation provide more value than a single agent could?"),
    ),
    reward_text_override=_LEGACY_GROUP_TEXT,
)


_RUBRICS: Dict[Tuple[str, str], Rubric] = {
    ("individual", PAPER_V1): INDIVIDUAL_RUBRIC,
    ("group", PAPER_V1): GROUP_RUBRIC,
    ("individual", LEGACY_V0): INDIVIDUAL_RUBRIC_LEGACY,
    ("group", LEGACY_V0): GROUP_RUBRIC_LEGACY,
}


def get_rubric(kind: str, version: str = PAPER_V1) -> Rubric:
    """Look up a rubric by kind ('individual'|'group') and version."""
    try:
        return _RUBRICS[(kind, version)]
    except KeyError:
        raise ValueError(
            f"Unknown rubric kind={kind!r} version={version!r}; "
            f"kinds: individual|group, versions: {RUBRIC_VERSIONS}") from None


def format_rubric(rubric: Rubric, style: str = "judge") -> str:
    """Render a rubric as prompt text.

    style='judge': numbered verbatim judge blocks + epilogue (paper_v1 only;
        byte-identical to the historical judge prompts).
    style='reward_prompt': criteria text for reward-model prompts. legacy_v0
        returns the exact string existing checkpoints were trained with.
    """
    if style == "judge":
        if any(not c.judge_block for c in rubric.criteria):
            raise ValueError(
                f"Rubric {rubric.kind}/{rubric.version} has no judge blocks "
                "(legacy rubrics are reward-prompt only)")
        return ("\n"
                + "".join(f"{i + 1}. {c.judge_block}" for i, c in enumerate(rubric.criteria))
                + rubric.epilogue)
    if style == "reward_prompt":
        if rubric.reward_text_override:
            return rubric.reward_text_override
        return ("\n"
                + "\n".join(f"{i + 1}. {c.title}: {c.description}"
                            for i, c in enumerate(rubric.criteria))
                + "\n")
    raise ValueError(f"Unknown rubric style {style!r}; expected 'judge' or 'reward_prompt'")


def validate_scores(scores: Dict, rubric: Rubric, strict: bool = True) -> Dict[str, list]:
    """Check that a judge-returned score dict matches the rubric schema.

    Returns {'missing': [...], 'unexpected': [...]}. In strict mode a
    non-empty mismatch raises ValueError.
    """
    expected = set(rubric.keys())
    got = set(scores)
    mismatch = {"missing": sorted(expected - got), "unexpected": sorted(got - expected)}
    if strict and (mismatch["missing"] or mismatch["unexpected"]):
        raise ValueError(
            f"Score keys do not match {rubric.kind}/{rubric.version} rubric: "
            f"missing={mismatch['missing']}, unexpected={mismatch['unexpected']}")
    return mismatch


# ---------------------------------------------------------------------------
# Reward-model checkpoint metadata
# ---------------------------------------------------------------------------

def write_checkpoint_metadata(checkpoint_dir: str, *, base_model: str,
                              reward_type: str, rubric_version: str) -> str:
    """Record base model, reward type (persona|group), and rubric version."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, CHECKPOINT_METADATA_FILENAME)
    with open(path, "w") as f:
        json.dump({
            "schema_version": "1.0",
            "base_model": base_model,
            "reward_type": reward_type,
            "rubric_version": rubric_version,
        }, f, indent=2)
    return path


def read_checkpoint_metadata(checkpoint_dir: str) -> Optional[Dict]:
    path = os.path.join(checkpoint_dir, CHECKPOINT_METADATA_FILENAME)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def validate_dataset_rubric_version(dataset, requested: Optional[str] = None,
                                    assume_legacy: bool = False,
                                    source: str = "dataset") -> str:
    """Validate a training dataset's rubric_version before reward-model training.

    ``dataset`` is an HF Dataset (or any sequence of row dicts). Rules:
    - mixed rubric_version values across rows → hard error;
    - a uniform recorded version that contradicts an explicitly requested one
      → hard error (never silently train with a mismatched prompt schema);
    - rows without rubric_version metadata require an explicit
      ``--rubric-version`` or ``--assume-legacy-rubric`` (never assumed).

    Returns the resolved rubric version to train with.
    """
    column_names = getattr(dataset, "column_names", None)
    if column_names is not None:
        values = set(dataset["rubric_version"]) if "rubric_version" in column_names else set()
    else:
        values = {row.get("rubric_version") for row in dataset} - {None}

    if len(values) > 1:
        raise ValueError(
            f"{source} mixes rubric versions {sorted(values)}; reward-model "
            "training requires a single rubric schema — regenerate or split the data.")
    if values:
        recorded = values.pop()
        if recorded not in RUBRIC_VERSIONS:
            raise ValueError(f"{source} records unknown rubric_version={recorded!r}")
        if requested is not None and requested != recorded:
            raise ValueError(
                f"{source} was annotated with rubric_version={recorded!r} but "
                f"--rubric-version {requested!r} was given. Refusing to train "
                "with a mismatched prompt schema.")
        return recorded
    if requested is not None:
        return requested
    if assume_legacy:
        return LEGACY_V0
    raise ValueError(
        f"{source} carries no rubric_version metadata. Pass --rubric-version "
        f"{{{','.join(RUBRIC_VERSIONS)}}} or --assume-legacy-rubric explicitly "
        "(never silently assumed).")


def resolve_checkpoint_rubric_version(checkpoint_dir: str,
                                      cli_version: Optional[str] = None,
                                      assume_legacy: bool = False) -> str:
    """Determine which rubric version to use with a reward-model checkpoint.

    Precedence: recorded checkpoint metadata > explicit CLI version >
    --assume-legacy-rubric. A checkpoint with no recorded version and no
    explicit choice is a hard error — never silently assumed.
    If both metadata and an explicit CLI version exist and disagree, that is
    also a hard error (never silently score with a mismatched prompt schema).
    """
    meta = read_checkpoint_metadata(checkpoint_dir)
    recorded = meta.get("rubric_version") if meta else None
    if recorded is not None and recorded not in RUBRIC_VERSIONS:
        raise ValueError(
            f"Checkpoint {checkpoint_dir} records unknown rubric_version={recorded!r}")
    if recorded is not None:
        if cli_version is not None and cli_version != recorded:
            raise ValueError(
                f"Checkpoint {checkpoint_dir} was trained with rubric_version="
                f"{recorded!r} but --rubric-version {cli_version!r} was given. "
                "Refusing to score with a mismatched prompt schema.")
        return recorded
    if cli_version is not None:
        return cli_version
    if assume_legacy:
        return LEGACY_V0
    raise ValueError(
        f"Reward checkpoint {checkpoint_dir} has no recorded rubric version "
        f"({CHECKPOINT_METADATA_FILENAME} missing). Pass --rubric-version "
        f"{{{','.join(RUBRIC_VERSIONS)}}} or --assume-legacy-rubric explicitly.")
