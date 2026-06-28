"""GPU pool scheduler for parallel experiment execution.

Dispatches jobs to a pool of GPU *groups*, automatically assigning
CUDA_VISIBLE_DEVICES (comma-joined per group), MASTER_PORT, and NPROC_PER_NODE.
Each group is one slot in the pool: a 1-GPU group runs single-process jobs;
a multi-GPU group runs DDP-style jobs across the group's GPUs.

Both ``list[int]`` and ``list[list[int]]`` are accepted; the former is
auto-promoted to one-GPU-per-slot for backwards compatibility.

Usage (one GPU per slot):
    from gpu_pool import GPUPool, Job
    pool = GPUPool([0, 1, 2, 3])             # 4 single-GPU slots
    jobs = [Job(name="train_mbpp", cmd=["python", "train.py", ...], log_file="logs/train.log")]
    failed = pool.run(jobs, label="Train")

Usage (multi-GPU groups + dependencies):
    from gpu_pool import JobNode, JobGraph, DependencyScheduler
    graph = JobGraph()
    graph.add(JobNode(name="train", cmd_factory=lambda: ["torchrun", "--nproc_per_node",
                                                          os.environ["NPROC_PER_NODE"], ...],
                      phase="train"))
    graph.add(JobNode(name="eval", cmd=[...], depends_on=["train"], phase="eval"))
    scheduler = DependencyScheduler([[0, 1], [2, 3]])   # 2 dual-GPU slots
    failed = scheduler.run(graph)
"""

import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Callable
from zoneinfo import ZoneInfo

_failed_cmd_lock = Lock()


def parse_gpu_groups(spec: str) -> list[list[int]]:
    """Parse "0,1 2,3" -> [[0,1],[2,3]]; "0 1 2 3" -> [[0],[1],[2],[3]]."""
    groups = []
    for token in spec.split():
        gpus = [int(x) for x in token.split(",") if x.strip()]
        if not gpus:
            raise ValueError(f"empty GPU group in --gpu-groups: {spec!r}")
        groups.append(gpus)
    if not groups:
        raise ValueError(f"--gpu-groups must specify at least one group: {spec!r}")
    return groups


def _normalize_gpu_groups(gpu_groups) -> list[list[int]]:
    """Accept either ``list[int]`` (legacy: one GPU per slot) or ``list[list[int]]``
    (each inner list is a group of GPUs that share one job slot).
    """
    if not gpu_groups:
        return []
    if all(isinstance(g, int) for g in gpu_groups):
        return [[g] for g in gpu_groups]
    return [list(g) for g in gpu_groups]


def log_failed_cmd(cmd: list[str], file: str = "failed_commands.md"):
    """Print and append a failed command with ET timestamp to failed_commands.md."""
    cmd_str = " ".join(cmd)
    ts = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S ET")
    print(f"  FAILED CMD [{ts}]: {cmd_str}")
    with _failed_cmd_lock:
        with open(file, "a") as f:
            f.write(f"# Failed at {ts}\n{cmd_str}\n\n")


@dataclass
class Job:
    """A single GPU job to execute."""
    name: str
    cmd: list[str]
    env: dict = field(default_factory=dict)
    log_file: str | None = None
    metadata: dict = field(default_factory=dict)


class GPUPool:
    """Dispatch jobs across a pool of GPU groups (one job per group at a time)."""

    def __init__(self, gpu_groups, base_port: int = 29500):
        """gpu_groups: ``list[int]`` (one GPU per slot) or ``list[list[int]]``."""
        self.gpu_groups: list[list[int]] = _normalize_gpu_groups(gpu_groups)
        self.base_port = base_port
        self._running: dict[int, tuple[subprocess.Popen, Job, object]] = {}
        self._job_counter = 0

    @property
    def gpu_ids(self) -> list[int]:
        """Flat list of all GPUs across groups (for backwards-compat callers)."""
        return [g for group in self.gpu_groups for g in group]

    def run(self, jobs: list[Job], label: str = "", on_complete=None) -> list[Job]:
        """Run jobs across GPU pool. Returns list of failed jobs.

        Args:
            on_complete: Optional callback(job: Job) -> list[Job].
                         Called when a job succeeds; returned jobs are added to queue.
        """
        if not jobs:
            return []

        failed = []
        queue = list(jobs)
        completed = 0
        total = len(queue)

        print(f"\n{'=' * 60}")
        print(f"  {label}: {total} jobs on {len(self.gpu_groups)} GPU groups "
              f"({sum(len(g) for g in self.gpu_groups)} GPUs)")
        print(f"{'=' * 60}")

        try:
            # Fill initial batch
            for group_idx in range(len(self.gpu_groups)):
                if not queue:
                    break
                job = queue.pop(0)
                self._start_job(group_idx, job, label, total - len(queue), total)

            # Poll loop
            while self._running:
                for group_idx in list(self._running.keys()):
                    proc, job, log_fh = self._running[group_idx]
                    rc = proc.poll()
                    if rc is not None:
                        if log_fh:
                            log_fh.close()
                        del self._running[group_idx]
                        gpu_label = self._group_label(group_idx)
                        if rc != 0:
                            failed.append(job)
                            print(f"[{label}] {gpu_label} FAILED (rc={rc}): {job.name}")
                            log_failed_cmd(job.cmd)
                            self._print_log_tail(job.log_file)
                        else:
                            completed += 1
                            print(f"[{label}] {gpu_label} done ({completed}/{total}): {job.name}")

                            # Follow-up jobs from callback
                            if on_complete:
                                followup = on_complete(job)
                                if followup:
                                    queue.extend(followup)
                                    total += len(followup)

                        # Dispatch next job to freed group (small delay for port/memory release)
                        if queue:
                            time.sleep(5)
                            next_job = queue.pop(0)
                            self._start_job(group_idx, next_job, label, total - len(queue), total)

                if self._running:
                    time.sleep(3)

        except KeyboardInterrupt:
            print(f"\n[{label}] Interrupted — terminating {len(self._running)} running jobs...")
            self._terminate_all()
            sys.exit(1)

        print(f"[{label}] {completed}/{total} completed, {len(failed)} failed")
        return failed

    def run_dry(self, jobs: list[Job], label: str = ""):
        """Print jobs without executing."""
        if not jobs:
            print(f"  ({label}: no jobs)")
            return
        print(f"\n--- {label}: {len(jobs)} jobs ---")
        for i, job in enumerate(jobs, 1):
            print(f"  [{i}] {job.name}")
            print(f"      cmd: {' '.join(job.cmd)}")
            if job.log_file:
                print(f"      log: {job.log_file}")

    def _start_job(self, group_idx: int, job: Job, label: str, idx: int, total: int):
        group = self.gpu_groups[group_idx]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, group))
        env["MASTER_ADDR"] = "127.0.0.1"
        # Use unique port per job to avoid EADDRINUSE from rapid job cycling
        env["MASTER_PORT"] = str(self.base_port + self._job_counter * 10 + group_idx)
        env["NPROC_PER_NODE"] = str(len(group))
        self._job_counter += 1
        env.update(job.env)

        log_fh = None
        if job.log_file:
            os.makedirs(os.path.dirname(job.log_file), exist_ok=True)
            log_fh = open(job.log_file, "w")
            proc = subprocess.Popen(job.cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        else:
            proc = subprocess.Popen(job.cmd, env=env)

        self._running[group_idx] = (proc, job, log_fh)
        print(f"[{label}] {self._group_label(group_idx)} <- Job {idx}/{total}: {job.name}")

    def _group_label(self, group_idx: int) -> str:
        group = self.gpu_groups[group_idx]
        if len(group) == 1:
            return f"GPU {group[0]}"
        return f"Group {group_idx} (GPUs {','.join(map(str, group))})"

    @staticmethod
    def _print_log_tail(log_file: str | None, n: int = 30):
        """Print last n lines of a log file to stderr."""
        if not log_file or not os.path.isfile(log_file):
            return
        try:
            with open(log_file) as f:
                lines = f.readlines()
            tail = lines[-n:]
            print(f"  ┌── last {len(tail)} lines of {log_file} ──", file=sys.stderr)
            for line in tail:
                print(f"  │ {line}", end="", file=sys.stderr)
            print(f"  └{'─' * 60}", file=sys.stderr)
        except OSError:
            pass

    def _terminate_all(self):
        for group_idx, (proc, job, log_fh) in list(self._running.items()):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            if log_fh:
                log_fh.close()
        self._running.clear()


# ══════════════════════════════════════════════════════════════════════════════
# Dependency-Aware Scheduler (new)
# ══════════════════════════════════════════════════════════════════════════════

# Status constants
PENDING = "pending"
BLOCKED = "blocked"
RUNNABLE = "runnable"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
SKIPPED = "skipped"
CANCELLED = "cancelled"

TERMINAL_STATUSES = {DONE, FAILED, SKIPPED, CANCELLED}
# Statuses that satisfy downstream dependencies
SATISFIES_DEPS = {DONE, SKIPPED}

# Phase priority for scheduling (lower = higher priority)
PHASE_PRIORITY = {"cache": 0, "train": 1, "eval": 2}


@dataclass
class JobNode:
    """A job with dependency tracking for the graph scheduler."""
    name: str
    cmd: list[str] = field(default_factory=list)
    cmd_factory: Callable[[], list[str]] | None = None
    env: dict = field(default_factory=dict)
    log_file: str | None = None
    metadata: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    status: str = PENDING
    phase: str = ""
    experiment: str = ""


class JobGraph:
    """Directed acyclic graph of JobNodes with dependency tracking."""

    def __init__(self):
        self._nodes: dict[str, JobNode] = {}
        self._dependents: dict[str, list[str]] = defaultdict(list)

    def add(self, node: JobNode):
        """Add a node. Deduplicates by name: same name = same logical job.

        When two builders produce the same-named node (e.g., build_induction and
        build_teacher_agreement both produce induction training nodes), we keep
        the first one added. Argument order may differ but the job is identical.
        """
        if node.name in self._nodes:
            return  # Silently deduplicate
        self._nodes[node.name] = node
        for dep in node.depends_on:
            self._dependents[dep].append(node.name)

    def add_all(self, nodes: list[JobNode]):
        for node in nodes:
            self.add(node)

    def resolve_initial_status(self):
        """Set initial status for all nodes: skipped stays, others become blocked or runnable."""
        for node in self._nodes.values():
            if node.status in TERMINAL_STATUSES:
                continue
            # Check if all deps are satisfied
            if self._deps_satisfied(node):
                node.status = RUNNABLE
            else:
                node.status = BLOCKED

    def _deps_satisfied(self, node: JobNode) -> bool:
        for dep_name in node.depends_on:
            if dep_name not in self._nodes:
                return False
            if self._nodes[dep_name].status not in SATISFIES_DEPS:
                return False
        return True

    def _deps_failed(self, node: JobNode) -> bool:
        """Check if any dependency has failed or been cancelled."""
        for dep_name in node.depends_on:
            if dep_name not in self._nodes:
                return True
            if self._nodes[dep_name].status in {FAILED, CANCELLED}:
                return True
        return False

    def get_runnable(self) -> list[JobNode]:
        """Get runnable nodes, prioritized by phase (cache > train > eval), then insertion order."""
        runnable = [n for n in self._nodes.values() if n.status == RUNNABLE]
        runnable.sort(key=lambda n: PHASE_PRIORITY.get(n.phase, 99))
        return runnable

    def mark_done(self, name: str) -> list[JobNode]:
        """Mark node done and return newly-runnable nodes."""
        node = self._nodes[name]
        node.status = DONE
        return self._unblock_dependents(name)

    def mark_failed(self, name: str) -> list[JobNode]:
        """Mark node failed, cancel dependents, return other newly-runnable nodes."""
        node = self._nodes[name]
        node.status = FAILED
        self._cancel_dependents(name)
        # Other nodes may have become runnable from unrelated completions
        return []

    def _unblock_dependents(self, name: str) -> list[JobNode]:
        """Check dependents of `name` and transition blocked→runnable if all deps satisfied."""
        newly_runnable = []
        for dep_name in self._dependents.get(name, []):
            dep_node = self._nodes[dep_name]
            if dep_node.status == BLOCKED and self._deps_satisfied(dep_node):
                dep_node.status = RUNNABLE
                newly_runnable.append(dep_node)
        return newly_runnable

    def _cancel_dependents(self, name: str):
        """Recursively cancel all dependents of a failed/cancelled node."""
        for dep_name in self._dependents.get(name, []):
            dep_node = self._nodes[dep_name]
            if dep_node.status not in TERMINAL_STATUSES:
                dep_node.status = CANCELLED
                self._cancel_dependents(dep_name)

    def summary(self) -> dict[str, int]:
        counts = defaultdict(int)
        for node in self._nodes.values():
            counts[node.status] += 1
        return dict(counts)

    def all_terminal(self) -> bool:
        return all(n.status in TERMINAL_STATUSES for n in self._nodes.values())

    def __len__(self):
        return len(self._nodes)

    def nodes(self) -> list[JobNode]:
        return list(self._nodes.values())


class DependencyScheduler:
    """Dispatches JobGraph nodes across GPU groups respecting dependencies."""

    def __init__(self, gpu_groups, base_port: int = 29500):
        """gpu_groups: ``list[int]`` (one GPU per slot) or ``list[list[int]]``."""
        self.gpu_groups: list[list[int]] = _normalize_gpu_groups(gpu_groups)
        self.base_port = base_port
        self._running: dict[int, tuple[subprocess.Popen, JobNode, object]] = {}

    @property
    def gpu_ids(self) -> list[int]:
        """Flat list of all GPUs across groups (for backwards-compat callers)."""
        return [g for group in self.gpu_groups for g in group]

    def _group_label(self, group_idx: int) -> str:
        group = self.gpu_groups[group_idx]
        if len(group) == 1:
            return f"GPU {group[0]}"
        return f"Group {group_idx} (GPUs {','.join(map(str, group))})"

    def run(self, graph: JobGraph) -> list[JobNode]:
        """Run all jobs in the graph respecting dependencies. Returns failed+cancelled nodes."""
        graph.resolve_initial_status()

        summary = graph.summary()
        total = len(graph)
        print(f"\n{'=' * 60}")
        print(f"  Dependency Scheduler: {total} jobs on {len(self.gpu_groups)} GPU groups "
              f"({sum(len(g) for g in self.gpu_groups)} GPUs)")
        print(f"  {self._format_summary(summary)}")
        print(f"{'=' * 60}")

        last_status_time = time.time()

        try:
            while not graph.all_terminal():
                # Dispatch to free GPU groups
                free_groups = [i for i in range(len(self.gpu_groups))
                               if i not in self._running]
                if free_groups:
                    runnable = graph.get_runnable()
                    for group_idx in free_groups:
                        if not runnable:
                            break
                        node = runnable.pop(0)
                        # Resolve cmd_factory if needed (after env is built so factories
                        # can read NPROC_PER_NODE; but env is set inside _start_job, so
                        # we expose group size to factories via a temporary env var here)
                        if node.cmd_factory and not node.cmd:
                            os.environ["NPROC_PER_NODE"] = str(len(self.gpu_groups[group_idx]))
                            try:
                                node.cmd = node.cmd_factory()
                            except Exception as e:
                                self._ts_print(f"{node.name}: cmd_factory failed: {e}")
                                graph.mark_failed(node.name)
                                cancelled = [n for n in graph.nodes()
                                             if n.status == CANCELLED]
                                if cancelled:
                                    self._ts_print(
                                        f"  cancelled {len(cancelled)} dependents")
                                continue
                            finally:
                                os.environ.pop("NPROC_PER_NODE", None)
                        node.status = RUNNING
                        self._start_job(group_idx, node)

                # Poll running processes
                for group_idx in list(self._running.keys()):
                    proc, node, log_fh = self._running[group_idx]
                    rc = proc.poll()
                    if rc is None:
                        continue

                    if log_fh:
                        log_fh.close()
                    del self._running[group_idx]

                    if rc != 0:
                        # Check for OOM and auto-retry with reduced batch size
                        if self._is_oom_failure(node.log_file):
                            new_bsz = self._reduce_batch_size(node.cmd)
                            if new_bsz is not None:
                                self._ts_print(
                                    f"{node.name}: OOM detected, retrying with "
                                    f"batch_size={new_bsz}")
                                node.status = RUNNABLE
                                continue

                        self._ts_print(f"{node.name}: FAILED (rc={rc}) on {self._group_label(group_idx)}")
                        if node.cmd:
                            log_failed_cmd(node.cmd)
                        GPUPool._print_log_tail(node.log_file)
                        graph.mark_failed(node.name)
                        cancelled = [n for n in graph.nodes()
                                     if n.status == CANCELLED]
                        if cancelled:
                            cancel_names = [n.name for n in cancelled]
                            self._ts_print(
                                f"  cancelled: {', '.join(cancel_names)}")
                    else:
                        newly_runnable = graph.mark_done(node.name)
                        msg = f"{node.name}: done on {self._group_label(group_idx)}"
                        if newly_runnable:
                            msg += f" -> unblocked: {', '.join(n.name for n in newly_runnable)}"
                        self._ts_print(msg)

                # Periodic status
                now = time.time()
                if now - last_status_time >= 30:
                    s = graph.summary()
                    self._ts_print(f"[Status] {self._format_summary(s)}")
                    free_groups = [i for i in range(len(self.gpu_groups))
                                   if i not in self._running]
                    if free_groups:
                        runnable_count = len(graph.get_runnable())
                        blocked_count = s.get(BLOCKED, 0)
                        running_count = s.get(RUNNING, 0)
                        if runnable_count == 0 and blocked_count > 0:
                            free_labels = ", ".join(self._group_label(i) for i in free_groups)
                            self._ts_print(
                                f"  {free_labels} idle — "
                                f"0 runnable, {blocked_count} blocked "
                                f"(waiting for {running_count} running)")
                    last_status_time = now

                if self._running:
                    time.sleep(3)

        except KeyboardInterrupt:
            print(f"\nInterrupted — terminating {len(self._running)} running jobs...")
            self._terminate_all()
            sys.exit(1)

        # Final report
        self._print_final_report(graph)

        return [n for n in graph.nodes()
                if n.status in {FAILED, CANCELLED}]

    def run_dry(self, graph: JobGraph):
        """Print layered plan without executing."""
        graph.resolve_initial_status()

        summary = graph.summary()
        total = len(graph)
        to_run = summary.get(RUNNABLE, 0) + summary.get(BLOCKED, 0)
        skipped = summary.get(SKIPPED, 0)

        print(f"\n{'=' * 60}")
        print(f"  DRY RUN: {total} total jobs ({to_run} to run, {skipped} skipped)")
        print(f"  GPU groups: {len(self.gpu_groups)} "
              f"(slots: {[','.join(map(str, g)) for g in self.gpu_groups]})")
        print(f"{'=' * 60}")

        # Group by topological depth
        depths = self._compute_depths(graph)
        max_depth = max(depths.values()) if depths else 0

        for depth in range(max_depth + 1):
            nodes_at_depth = [n for n in graph.nodes()
                              if depths.get(n.name, 0) == depth]
            if not nodes_at_depth:
                continue

            print(f"\n  Layer {depth}:")
            for node in nodes_at_depth:
                if node.status == SKIPPED:
                    tag = "[SKIP]"
                elif node.status == RUNNABLE:
                    tag = "[RUN] "
                else:
                    tag = "[WAIT]"

                deps_str = ""
                if node.depends_on:
                    deps_str = f" (deps: {', '.join(node.depends_on)})"

                print(f"    {tag} {node.name}{deps_str}")
                if node.status != SKIPPED and node.cmd:
                    print(f"           cmd: {' '.join(node.cmd[:6])}...")
                elif node.status != SKIPPED and node.cmd_factory:
                    print(f"           cmd: <deferred>")

        print()

    def _compute_depths(self, graph: JobGraph) -> dict[str, int]:
        """Compute topological depth for each node."""
        depths = {}
        def _depth(name):
            if name in depths:
                return depths[name]
            node = graph._nodes.get(name)
            if not node or not node.depends_on:
                depths[name] = 0
                return 0
            d = max(_depth(dep) for dep in node.depends_on if dep in graph._nodes) + 1
            depths[name] = d
            return d
        for name in graph._nodes:
            _depth(name)
        return depths

    def _start_job(self, group_idx: int, node: JobNode):
        group = self.gpu_groups[group_idx]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, group))
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = str(self.base_port + group_idx * 10)
        env["NPROC_PER_NODE"] = str(len(group))
        env.update(node.env)

        log_fh = None
        if node.log_file:
            os.makedirs(os.path.dirname(node.log_file), exist_ok=True)
            log_fh = open(node.log_file, "w")
            proc = subprocess.Popen(node.cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        else:
            proc = subprocess.Popen(node.cmd, env=env)

        self._running[group_idx] = (proc, node, log_fh)
        self._ts_print(f"{self._group_label(group_idx)} <- {node.name}")

    def _terminate_all(self):
        for group_idx, (proc, node, log_fh) in list(self._running.items()):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            if log_fh:
                log_fh.close()
        self._running.clear()

    @staticmethod
    def _ts_print(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")

    @staticmethod
    def _is_oom_failure(log_file: str | None) -> bool:
        """Check if a failure was due to CUDA OOM."""
        if not log_file or not os.path.isfile(log_file):
            return False
        try:
            with open(log_file) as f:
                content = f.read()
            return "CUDA out of memory" in content or "OutOfMemoryError" in content
        except OSError:
            return False

    @staticmethod
    def _reduce_batch_size(cmd: list[str]) -> int | None:
        """Reduce --per_device_train_batch_size in cmd. Returns new value or None if can't reduce."""
        try:
            idx = cmd.index("--per_device_train_batch_size")
        except ValueError:
            return None
        current = int(cmd[idx + 1])
        if current <= 1:
            return None
        new_bsz = current - 1 if current <= 4 else current // 2
        cmd[idx + 1] = str(new_bsz)
        return new_bsz

    @staticmethod
    def _format_summary(s: dict[str, int]) -> str:
        parts = []
        for status in [DONE, RUNNING, RUNNABLE, BLOCKED, SKIPPED, CANCELLED, FAILED]:
            if s.get(status, 0) > 0:
                parts.append(f"{status}={s[status]}")
        return " ".join(parts)

    @staticmethod
    def _print_final_report(graph: JobGraph):
        s = graph.summary()
        total = len(graph)
        print(f"\n{'=' * 60}")
        print(f"  Final Report")
        print(f"{'=' * 60}")
        print(f"  Total: {total}  Done: {s.get(DONE, 0)}  Skipped: {s.get(SKIPPED, 0)}  "
              f"Failed: {s.get(FAILED, 0)}  Cancelled: {s.get(CANCELLED, 0)}")

        failed_nodes = [n for n in graph.nodes() if n.status == FAILED]
        if failed_nodes:
            print(f"\n  Failed:")
            for n in failed_nodes:
                log_info = f"  (log: {n.log_file})" if n.log_file else ""
                print(f"    {n.name}{log_info}")

        cancelled_nodes = [n for n in graph.nodes() if n.status == CANCELLED]
        if cancelled_nodes:
            print(f"\n  Cancelled (dependency failed):")
            for n in cancelled_nodes:
                blocked_by = [d for d in n.depends_on
                              if d in graph._nodes and graph._nodes[d].status in {FAILED, CANCELLED}]
                print(f"    {n.name}  (blocked by: {', '.join(blocked_by)})")

        if not failed_nodes and not cancelled_nodes:
            print(f"\n  All jobs completed successfully.")
        print(f"{'=' * 60}")
