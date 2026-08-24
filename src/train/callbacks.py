"""Shared trainer callbacks and run instrumentation (Phase-2 Part B).

RewardLoggingCallback moved here from src/train/train_rl.py so the group
trainers no longer import implementation details from the Speaker trainer;
train_rl re-exports it for backward compatibility. This module must not
import train_rl or train_group.
"""

import logging
import os

from transformers import TrainerCallback

try:
    import wandb
except ImportError:  # wandb is an optional dependency
    wandb = None

logger = logging.getLogger(__name__)


class RewardLoggingCallback(TrainerCallback):
    """Log per-component reward breakdown every N steps (rank 0 only)."""

    def __init__(self, reward_names, reward_weights, every_n_steps=50):
        self.reward_names = reward_names
        self.reward_weights = reward_weights
        self.every_n_steps = every_n_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return
        step = state.global_step
        if step == 0 or step % self.every_n_steps != 0:
            return

        lines = [f"[Step {step}] Reward breakdown:"]
        weighted_mean = 0.0
        wb_metrics = {}

        for name, w in zip(self.reward_names, self.reward_weights):
            mean_key = f"rewards/{name}/mean"
            std_key = f"rewards/{name}/std"
            m = logs.get(mean_key)
            s = logs.get(std_key, 0.0)
            if m is not None:
                lines.append(f"  {name:25s}  w={w:.2f}  mean={m:+.4f}  std={s:.4f}")
                weighted_mean += w * m
                wb_metrics[f"reward_breakdown/{name}_weighted"] = w * m

        trl_combined = logs.get("reward", None)
        lines.append(f"  {'weighted_sum':25s}         mean={weighted_mean:+.4f}")
        if trl_combined is not None:
            diff = weighted_mean - trl_combined
            lines.append(f"  {'trl_combined':25s}         mean={trl_combined:+.4f}  diff={diff:+.4f}")
            wb_metrics["reward_breakdown/reconstructed"] = weighted_mean
            wb_metrics["reward_breakdown/trl_combined"] = trl_combined
            wb_metrics["reward_breakdown/recon_minus_trl"] = diff

        logger.info("\n".join(lines))

        if wandb is not None and wandb.run is not None and wb_metrics:
            wandb.log(wb_metrics, step=step)


def maybe_init_wandb_group(mode: str) -> None:
    """Initialize W&B for a group-training run when WANDB_PROJECT is set
    (same behavior as the historical train_group.main preamble)."""
    if os.getenv("WANDB_PROJECT"):
        import wandb as _wandb
        _wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            name=f"group-{mode}",
        )
        logger.info("W&B tracking enabled")
