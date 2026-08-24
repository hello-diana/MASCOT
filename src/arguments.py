import warnings
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter

from src.const import (
    DATASETS,
    DEFAULT_DIRECTOR_MODEL,
    DEFAULT_SEED,
    DEFAULT_SPEAKER_MODEL,
    DEFAULT_USER_MODEL,
    QMSUM_DATASET,
)


def add_server_args(parser) -> None:
    """Role-specific vLLM endpoint arguments (shared across CLIs).

    All defaults are None: a vLLM backend with no applicable URL either loads
    the model locally in-process (eval pipelines) or is a hard CLI error
    (server-only pipelines). Localhost URLs are never baked in as defaults —
    shell scripts pass them explicitly.
    """
    parser.add_argument(
        "--vllm-server-url", type=str, default=None,
        help="Shared external vLLM server URL (e.g., http://localhost:8008/v1) "
             "used for Director, Speaker, and User unless a role-specific URL "
             "overrides it. The judge endpoint never inherits this URL."
    )
    parser.add_argument(
        "--director-server-url", type=str, default=None,
        help="vLLM server URL for the Director (overrides --vllm-server-url)"
    )
    parser.add_argument(
        "--speaker-server-url", type=str, default=None,
        help="vLLM server URL for the Speaker (overrides --vllm-server-url)"
    )
    parser.add_argument(
        "--user-server-url", type=str, default=None,
        help="vLLM server URL for the simulated User (overrides --vllm-server-url)"
    )
    parser.add_argument(
        "--judge-server-url", type=str, default=None,
        help="vLLM server URL for the judge (independent; does not inherit "
             "--vllm-server-url)"
    )


def resolve_server_urls(args: Namespace) -> dict:
    """Collect role→URL mapping consumed by src.models.llm_models.get_client."""
    return {
        "director": getattr(args, "director_server_url", None),
        "speaker": getattr(args, "speaker_server_url", None),
        "user": getattr(args, "user_server_url", None),
        "judge": getattr(args, "judge_server_url", None),
        "shared": getattr(args, "vllm_server_url", None),
    }


def resolve_deprecated(args: Namespace, parser, mapping: dict,
                       new_defaults: dict = None) -> Namespace:
    """Resolve deprecated CLI aliases.

    ``mapping`` maps old argparse dest → new dest. If only the old flag was
    set, warn and copy it to the new dest. If both were set to different
    values, exit with a parser error (non-zero).

    ``new_defaults`` maps a new dest to its parser default for flags whose
    default is not None: a new-flag value equal to its default is treated as
    "not explicitly set", so passing only the deprecated alias still works.
    """
    new_defaults = new_defaults or {}
    for old, new in mapping.items():
        old_val = getattr(args, old, None)
        new_val = getattr(args, new, None)
        if new in new_defaults and new_val == new_defaults[new]:
            new_val = None
        if old_val is None:
            continue
        old_flag = "--" + old.replace("_", "-")
        new_flag = "--" + new.replace("_", "-")
        if new_val is not None and new_val != old_val:
            parser.error(f"Conflicting values for {new_flag}={new_val!r} and its "
                         f"deprecated alias {old_flag}={old_val!r}; pass only {new_flag}.")
        if new_val is None:
            warnings.warn(f"{old_flag} is deprecated; use {new_flag}", DeprecationWarning,
                          stacklevel=2)
            setattr(args, new, old_val)
    return args


def build_parser() -> ArgumentParser:
    """Build the shared MASCOT eval ArgumentParser without parsing."""
    parser = ArgumentParser(
        description="MASCOT evaluation pipeline with directive-driven agent selection",
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    # Dataset arguments
    dataset_group = parser.add_argument_group('Dataset')
    dataset_group.add_argument(
        "--dataset", type=str, default=QMSUM_DATASET, choices=list(DATASETS),
        help="Canonical dataset name from const.py (e.g. QMSum, Empathetic, ESConv). "
             "HuggingFace path resolved via DATASET2HF."
    )
    dataset_group.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "valid", "test"],
        help="Dataset split to use"
    )
    dataset_group.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of samples to process (for testing)"
    )
    dataset_group.add_argument(
        "--turns-per-segment", type=int, default=100,
        help="Number of transcript turns per meeting segment (QMSum)"
    )
    dataset_group.add_argument(
        "--local-path", type=str, default=None,
        help="Local path to dataset files (alternative to HF)"
    )

    # User simulation arguments (Empathetic)
    sim_group = parser.add_argument_group('User Simulation')
    sim_group.add_argument(
        "--enable-user-simulation", action="store_true",
        help="Enable user simulation (reactive mode)"
    )
    sim_group.add_argument(
        "--user-response-probability", type=float, default=0.5,
        help="Probability of user responding after each agent turn (0.0-1.0)"
    )
    sim_group.add_argument(
        "--enable-big-five", action="store_true",
        help="Enable Big Five (OCEAN) personality profiles for user simulation"
    )
    sim_group.add_argument(
        "--user-big-five-profile", type=str, nargs='+', default=None,
        help="Specific Big Five profile(s) to test (requires --enable-big-five). "
             "Example: --user-big-five-profile 'High Openness' 'Low Neuroticism'"
    )

    # Model arguments
    model_group = parser.add_argument_group('Models')
    model_group.add_argument(
        "--directive-model", type=str, default=DEFAULT_DIRECTOR_MODEL,
        help="Director (directive) model name or path"
    )
    model_group.add_argument(
        "--agent-model", type=str, default=DEFAULT_SPEAKER_MODEL,
        help="Speaker (agent) model name or path"
    )
    model_group.add_argument(
        "--user-model", type=str, default=DEFAULT_USER_MODEL,
        help="User simulator model name or path (only used if --enable-user-simulation)"
    )
    model_group.add_argument(
        "--directive-lora-path", type=str, default=None,
        help="Path to directive model LoRA adapters"
    )
    model_group.add_argument(
        "--agent-lora-path", type=str, default=None,
        help="Path to agent model LoRA adapters"
    )
    model_group.add_argument(
        "--user-lora-path", type=str, default=None,
        help="Path to user simulator LoRA adapters"
    )
    add_server_args(model_group)
    model_group.add_argument(
        "--share-model", action="store_true",
        help="Load a single vLLM model (configured via --agent-model/--agent-lora-path) "
             "and reuse it for the directive generator. Ignores --directive-model and "
             "--directive-lora-path."
    )

    # Generation arguments
    gen_group = parser.add_argument_group('Generation')
    gen_group.add_argument(
        "--num-turns", type=int, default=10,
        help="Number of conversation turns to generate"
    )
    gen_group.add_argument(
        "--directive-temperature", type=float, default=0.9,
        help="Sampling temperature for directive generation"
    )
    gen_group.add_argument(
        "--agent-temperature", type=float, default=0.9,
        help="Sampling temperature for agent response generation"
    )
    gen_group.add_argument(
        "--user-temperature", type=float, default=0.9,
        help="Sampling temperature for user simulation"
    )
    gen_group.add_argument(
        "--directive-max-tokens", type=int, default=4096,
        help="Max tokens for directive generation"
    )
    gen_group.add_argument(
        "--agent-max-tokens", type=int, default=4096,
        help="Max tokens for agent response generation"
    )
    gen_group.add_argument(
        "--user-max-tokens", type=int, default=2048,
        help="Max tokens for user simulation"
    )
    gen_group.add_argument(
        "--n-shot", type=int, default=3,
        help="Number of few-shot examples in agent prompts (0 = zero-shot)"
    )
    gen_group.add_argument(
        "--no-thinking", action="store_true",
        help="Disable <think></think> tags in generation"
    )

    # Infrastructure arguments
    infra_group = parser.add_argument_group('Infrastructure')
    infra_group.add_argument(
        "--directive-gpu-memory", type=float, default=0.47,
        help="GPU memory utilization for directive model (0.0-1.0)"
    )
    infra_group.add_argument(
        "--agent-gpu-memory", type=float, default=0.47,
        help="GPU memory utilization for agent model (0.0-1.0)"
    )
    infra_group.add_argument(
        "--user-gpu-memory", type=float, default=0.3,
        help="GPU memory utilization for user simulator model (0.0-1.0)"
    )
    infra_group.add_argument(
        "--directive-tensor-parallel-size", type=int, default=1,
        help="Number of GPUs for directive model tensor parallelism"
    )
    infra_group.add_argument(
        "--agent-tensor-parallel-size", type=int, default=1,
        help="Number of GPUs for agent model tensor parallelism"
    )
    infra_group.add_argument(
        "--user-tensor-parallel-size", type=int, default=1,
        help="Number of GPUs for user simulator tensor parallelism"
    )

    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Output root; eval results land in <output-dir>/mascot_eval/<dataset>/<split>"
    )
    output_group.add_argument(
        "--suffix", type=str, default="",
        help="Optional suffix for output filename"
    )
    output_group.add_argument(
        "--overwrite", action="store_true",
        help="Replace an existing output file for this configuration. Default: "
             "refuse to run when the output exists. (There is no --resume for "
             "episode generation: a run writes its whole output atomically at "
             "the end, so partial results cannot be continued.)"
    )

    # Misc arguments
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--strict", action="store_true",
        help="Terminate on the first generation failure, malformed directive, "
             "or empty response. Default (tolerant) mode records failure "
             "counters in run_summary.json and continues."
    )
    parser.add_argument(
        "--trace-role-calls", action="store_true",
        help="Record every generation request (role, requested model id, "
             "LoRA-alias flag, completion tokens, success) to "
             "role_call_trace.jsonl in the run metadata directory. "
             "Server-backend requests only."
    )
    parser.add_argument(
        "--mode", type=str, default="mascot",
        choices=["mascot", "mascot-g", "mascot-i"],
        help="Evaluation mode: 'mascot' (full), 'mascot-g' (random selection), 'mascot-i' (base model)"
    )

    return parser


def parse_arguments() -> Namespace:
    """Parse command-line arguments for MASCOT evaluation pipelines (QMSum and Empathetic)."""
    parser = build_parser()
    args = parser.parse_args()

    return args
