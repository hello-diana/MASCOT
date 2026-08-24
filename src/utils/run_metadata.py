"""Run-level configuration/summary serialization and failure counters.

Every generation/evaluation run writes two files into its output directory:

- ``run_config.json``  — immutable resolved startup configuration
  (schema version, run id, dataset, system, persona set, model roles, rubric
  version, resolved CLI args, seeds, GPU assignment).
- ``run_summary.json`` — runtime results written at completion (completion
  counts, failure/fallback counters, timing, final status).

Secrets are redacted from both files before serialization.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

RUN_CONFIG_FILENAME = "run_config.json"
RUN_SUMMARY_FILENAME = "run_summary.json"

# Tolerant-mode failure counters (spec I). Strict mode terminates instead.
COUNTER_KEYS = (
    "num_generation_failures",
    "num_directive_parse_failures",
    "num_random_agent_fallbacks",
    "num_empty_responses",
    "num_annotation_failures",
    "num_skipped_samples",
)

# Credential-aware KEY classifier.
#
# We must NOT redact reproducibility fields that merely contain the substring
# "token" (max_tokens, prompt_tokens, completion_tokens, num_tokens,
# token_count, tokenizer, ...). So "token" is only treated as a secret when it
# carries a credential qualifier (access_/auth_/bearer_/hf_/refresh_/session_/
# id_token) or is the whole key. The remaining credential words (api_key,
# secret, password, authorization, credential, wandb_key, private/client key,
# ...) are matched as whole tokens delimited by string start/end or a
# separator, so "secretariat" or "keyframe" do not trip them.
_CREDENTIAL_KEY_RE = re.compile(
    r"""(?ix)
    (?:^|[_\-])
    (?:
        api[_-]?key
      | (?:openai|anthropic|wandb|hf|huggingface|azure|aws|gcp)[_-]?api[_-]?key
      | wandb[_-]?key
      | secret | client[_-]?secret | api[_-]?secret
      | password | passwd
      | credentials?
      | authorization
      | (?:access|auth|bearer|refresh|session|id|hf|huggingface)[_-]?token
      | (?:private|secret|client)[_-]?key
    )
    (?:$|[_\-])
    """)
# Bare keys that are credentials on their own (not compound count/config names).
_EXACT_CREDENTIAL_KEYS = frozenset({"token", "auth", "apikey"})

# Common credential VALUE shapes: OpenAI keys, AWS access keys, HF tokens,
# bearer headers, and user:pass@ URLs.
_SECRET_VALUE_RE = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|AKIA[A-Z0-9]{12,}|hf_[A-Za-z0-9]{16,}"
    r"|Bearer\s+\S+|://[^/\s:]+:[^@/\s]+@)")

REDACTED = "[REDACTED]"


def _is_secret_key(key: Any) -> bool:
    """True only for keys that name an actual credential.

    Credential-bearing keys (api_key, hf_token, wandb_key, authorization,
    password, client_secret, ...) are redacted; ordinary token-budget /
    tokenizer fields (max_tokens, completion_tokens, num_tokens, token_count,
    tokenizer) are NOT.
    """
    k = str(key).lower()
    if k in _EXACT_CREDENTIAL_KEYS:
        return True
    return bool(_CREDENTIAL_KEY_RE.search(k))


def redact_secrets(obj: Any) -> Any:
    """Recursively redact credential-bearing keys and values."""
    if isinstance(obj, dict):
        return {
            k: (REDACTED if _is_secret_key(k) else redact_secrets(v))
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [redact_secrets(v) for v in obj]
    if isinstance(obj, str) and _SECRET_VALUE_RE.search(obj):
        return _SECRET_VALUE_RE.sub(REDACTED, obj)
    return obj


def new_counters() -> Dict[str, int]:
    return {key: 0 for key in COUNTER_KEYS}


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


_ENVIRONMENT_PACKAGES = ("torch", "transformers", "trl", "vllm", "peft",
                         "accelerate", "datasets", "openai", "pydantic")


def environment_versions() -> Dict[str, str]:
    """Exact versions of the key packages (plus python) for run_config."""
    import importlib.metadata as md
    import platform
    versions = {"python": platform.python_version()}
    for pkg in _ENVIRONMENT_PACKAGES:
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def model_revision(model_name: str) -> str:
    """Best-effort HF commit hash for a model resolved from the local hub cache.

    Returns "unknown" honestly when the model is a local path or the cache
    layout doesn't expose a revision — never fabricated.
    """
    try:
        from huggingface_hub import try_to_load_from_cache  # noqa: F401
        from huggingface_hub.constants import HF_HUB_CACHE
        repo_dir = Path(HF_HUB_CACHE) / f"models--{model_name.replace('/', '--')}"
        ref_main = repo_dir / "refs" / "main"
        if ref_main.is_file():
            return ref_main.read_text().strip()
    except Exception:
        pass
    return "unknown"


def new_run_id(prefix: str) -> str:
    """Unique, sortable, human-readable run id: {prefix}_{ts}_{uuid6}."""
    import uuid
    return f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"


def write_run_config(output_dir, *, run_id: str, dataset: Optional[str] = None,
                     system: Optional[str] = None, persona_set: Optional[str] = None,
                     model_roles: Optional[Dict[str, str]] = None,
                     rubric_version: Optional[str] = None,
                     args: Any = None, seed: Optional[int] = None,
                     extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write the immutable resolved startup configuration for a run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "dataset": dataset,
        "system": system,
        "persona_set": persona_set,
        "model_roles": model_roles or {},
        "model_revisions": {role: model_revision(name)
                            for role, name in (model_roles or {}).items() if name},
        "rubric_version": rubric_version,
        "seed": seed,
        "gpu_assignment": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "environment": environment_versions(),
        "resolved_args": {k: _jsonable(v) for k, v in sorted(vars(args).items())} if args else {},
    }
    if extra:
        config.update({k: _jsonable(v) for k, v in extra.items()})
    path = output_dir / RUN_CONFIG_FILENAME
    with open(path, "w") as f:
        json.dump(redact_secrets(config), f, indent=2)
    logger.info(f"Wrote run config to {path}")
    return path


def write_run_summary(output_dir, *, run_id: str, status: str,
                      counters: Optional[Dict[str, int]] = None,
                      started_at: Optional[float] = None,
                      extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write runtime results (counters, timing, final status) at completion."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "counters": {**new_counters(), **(counters or {})},
        "wall_time_seconds": round(time.time() - started_at, 2) if started_at else None,
    }
    if extra:
        summary.update({k: _jsonable(v) for k, v in extra.items()})
    path = output_dir / RUN_SUMMARY_FILENAME
    with open(path, "w") as f:
        json.dump(redact_secrets(summary), f, indent=2)
    logger.info(f"Wrote run summary to {path}")
    # Also surface the counters in the log, per the tolerant-mode contract.
    logger.info("Run summary counters: %s", summary["counters"])
    return path


def role_identity_metadata(identity: Dict[str, Dict],
                           requested: Dict[str, Optional[str]]):
    """Map live role-identity info (see llm_models.resolve_role_identity) to
    episode-metadata fields.

    Returns (base_models, adapter_extra): base_models holds
    director_model/speaker_model/user_model resolved to the BASE model id
    (falling back to the requested name for base/local clients); adapter_extra
    holds {role}_adapter fields ONLY for roles actually served via a LoRA
    alias — never fabricated for base-model runs.
    """
    base_models: Dict[str, Optional[str]] = {}
    adapter_extra: Dict[str, str] = {}
    for role in ("director", "speaker", "user"):
        info = identity.get(role) or {}
        base_models[f"{role}_model"] = info.get("base_model") or requested.get(role)
        if info.get("adapter"):
            adapter_extra[f"{role}_adapter"] = info["adapter"]
    return base_models, adapter_extra


def episode_record_metadata(*, system: str, dataset: str, persona_set: str,
                            director_model: str, speaker_model: str,
                            user_model: Optional[str] = None,
                            rubric_version: Optional[str] = None,
                            extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Canonical per-record metadata block for episode outputs.

    Only relevant fields are populated: ``user_model`` is omitted when no user
    simulation ran (never fabricated). Evaluation dispatches on this block
    instead of inferring the system from filenames.
    """
    meta = {
        "schema_version": SCHEMA_VERSION,
        "system": system,
        "dataset": dataset,
        "persona_set": persona_set,
        "director_model": director_model,
        "speaker_model": speaker_model,
    }
    if user_model is not None:
        meta["user_model"] = user_model
    if rubric_version is not None:
        meta["rubric_version"] = rubric_version
    if extra:
        meta.update(extra)
    return meta


class StrictModeError(RuntimeError):
    """Raised in --strict mode on the first generation/annotation failure."""


def record_failure(counters: Optional[Dict[str, int]], key: str,
                   strict: bool = False, detail: str = "") -> None:
    """Increment a failure counter; in strict mode terminate instead."""
    if counters is not None:
        counters[key] = counters.get(key, 0) + 1
    if strict:
        raise StrictModeError(f"--strict: {key}{': ' + detail if detail else ''}")
