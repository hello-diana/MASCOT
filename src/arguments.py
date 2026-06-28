from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter

from src.const import DATASETS, QMSUM_DATASET


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
        "--directive-model", type=str, default="Qwen/Qwen3-8B",
        help="Directive model name or path"
    )
    model_group.add_argument(
        "--agent-model", type=str, default="Qwen/Qwen3-8B",
        help="Agent model name or path"
    )
    model_group.add_argument(
        "--user-model", type=str, default="Qwen/Qwen3-8B",
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
    model_group.add_argument(
        "--vllm-server-url", type=str, default=None,
        help="URL of external vLLM server (e.g., http://localhost:8000/v1). "
             "When set, uses VLLMServerModel instead of loading local VLLMModel."
    )
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
        "--output-dir", type=str, default="outputs/mascot_eval",
        help="Output directory for results"
    )
    output_group.add_argument(
        "--suffix", type=str, default="",
        help="Optional suffix for output filename"
    )

    # Misc arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
