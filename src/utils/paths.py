"""Shared artifact-path construction (single source of truth).

Every stage of the MASCOT pipeline writes to / reads from directories built
here, so the README command chain connects without manual renames:

    generate individual candidates  -> data/rl/individual_{dataset}_{model}
    build preference pairs          -> (same dataset dir)
    train persona reward model      -> outputs/reward_model/Individual_{dataset}_{model}
    train Speaker GRPO              -> outputs/llm/grpo_Speaker_{dataset}_{model}
    generate group trajectories     -> data/rl/{group|branching}_{dataset}_{model}
    train group reward model        -> outputs/reward_model/group_reward_{dataset}_{model}
    train Director GRPO             -> outputs/llm/grpo_Director_{dataset}_{model}
    train Director DPO              -> outputs/llm/dpo_Director_{dataset}_{model}

The formats are byte-identical to the historical on-disk layout except:
- ``group_reward`` historically omitted the dataset (``group_reward_{model}``);
  loaders fall back to the legacy name with a warning (``legacy_dir``).
- ``director_grpo`` historically omitted the dataset (``grpo_Director_{model}``).
"""

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Preference/episode data modes (written by src.dataset.generate_and_annotate).
PREFERENCE_MODES = ("individual", "paired", "group", "branching")

# stage -> (root, name template). {mode} only used by preference_data.
_STAGE_TEMPLATES = {
    "preference_data": ("data/rl", "{mode}_{dataset}_{model}"),
    "persona_reward": ("outputs/reward_model", "Individual_{dataset}_{model}"),
    "group_reward": ("outputs/reward_model", "group_reward_{dataset}_{model}"),
    "speaker_grpo": ("outputs/llm", "grpo_Speaker_{dataset}_{model}"),
    "director_grpo": ("outputs/llm", "grpo_Director_{dataset}_{model}"),
    "director_dpo": ("outputs/llm", "dpo_Director_{dataset}_{model}"),
}

STAGES = tuple(_STAGE_TEMPLATES)


def model_short(model_name: str) -> str:
    """Short model tag used in artifact names, e.g. 'Qwen/Qwen3-8B' -> 'Qwen3-8B'."""
    return os.path.basename(str(model_name).rstrip("/"))


def artifact_dir(stage: str, *, dataset: str, model_name: str,
                 mode: str = None, split: str = None, root: str = None) -> Path:
    """Canonical artifact directory for one pipeline stage.

    Args:
        stage: One of ``STAGES``.
        dataset: Canonical dataset name (QMSum, Empathetic, ESConv).
        model_name: Full model id or path; shortened via ``model_short``.
        mode: Required for ``preference_data`` (one of ``PREFERENCE_MODES``).
        split: Optional split suffix, appended as ``_{split}``.
        root: Override the stage's default root directory.
    """
    if stage not in _STAGE_TEMPLATES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGES}")
    default_root, template = _STAGE_TEMPLATES[stage]
    if stage == "preference_data":
        if mode not in PREFERENCE_MODES:
            raise ValueError(
                f"preference_data requires mode in {PREFERENCE_MODES}, got {mode!r}")
    elif mode is not None:
        raise ValueError(f"mode is only valid for preference_data, not {stage!r}")

    name = template.format(mode=mode, dataset=dataset, model=model_short(model_name))
    if split:
        name += f"_{split}"
    return Path(root or default_root) / name


def legacy_dir(stage: str, *, dataset: str, model_name: str, root: str = None) -> Path:
    """Historical (pre-release) directory name for stages that were renamed.

    Only meaningful for ``group_reward`` (``group_reward_{model}``) and
    ``director_grpo`` (``grpo_Director_{model}``); other stages are unchanged.
    """
    short = model_short(model_name)
    if stage == "group_reward":
        return Path(root or "outputs/reward_model") / f"group_reward_{short}"
    if stage == "director_grpo":
        return Path(root or "outputs/llm") / f"grpo_Director_{short}"
    return artifact_dir(stage, dataset=dataset, model_name=model_name, root=root)


def resolve_artifact_dir(stage: str, *, dataset: str, model_name: str,
                         root: str = None) -> Path:
    """Return the canonical dir if it exists, else fall back to the legacy dir.

    Used on the LOAD side only: if the canonical path is missing but the
    legacy-named directory exists, warn and return the legacy path so existing
    user data keeps working. Writers always use ``artifact_dir``.
    """
    canonical = artifact_dir(stage, dataset=dataset, model_name=model_name, root=root)
    if canonical.exists():
        return canonical
    legacy = legacy_dir(stage, dataset=dataset, model_name=model_name, root=root)
    if legacy != canonical and legacy.exists():
        logger.warning(
            "Canonical path %s not found; falling back to legacy path %s. "
            "New runs will write to the canonical name.", canonical, legacy)
        return legacy
    return canonical


def file_fingerprint(path, max_content_bytes: int = 100_000_000) -> str:
    """Identity of a source data file: sha256 of its content (streamed), or a
    ``size:mtime_ns`` fallback for files larger than ``max_content_bytes``.
    """
    path = Path(path)
    stat = path.stat()
    if stat.st_size > max_content_bytes:
        return f"{stat.st_size}:{stat.st_mtime_ns}"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rollout_fingerprint(payload: dict) -> str:
    """12-hex digest of a complete rollout configuration (sorted-key JSON)."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def group_prefix_cache_path(dataset: str, speaker_model: str, num_turns: int,
                            fp12: str) -> Path:
    """Fingerprinted prefix-rollout cache dir. The fingerprint covers the full
    rollout config + source-data identity, so a cache can never be silently
    reused for a different configuration. Old pre-fingerprint cache dirs are
    never loaded.
    """
    return (Path("data/group_prefix")
            / f"{dataset}_{model_short(speaker_model)}_{num_turns}turns_{fp12}")


def ensure_writable_dir(path, overwrite: bool = False,
                        ignore: tuple = ("runs",)) -> Path:
    """Guard against silently clobbering an existing non-empty artifact dir.

    Entries named in ``ignore`` (default: the per-run ``runs/`` metadata dir)
    don't count toward "non-empty", so run metadata never blocks a fresh write.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        contents = [p for p in path.iterdir() if p.name not in ignore]
        if contents:
            raise FileExistsError(
                f"Output directory {path} exists and is not empty "
                f"(e.g. {contents[0].name}). Pass --overwrite to replace it"
                " (or --resume where the CLI supports resuming).")
    path.mkdir(parents=True, exist_ok=True)
    return path
