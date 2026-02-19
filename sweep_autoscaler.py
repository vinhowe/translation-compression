"""
Lightweight autoscaler for sweep_runner Slurm jobs.

Periodically scans sweep progress and submits Slurm jobs to maintain target
GPU counts across multiple partition profiles (H100, B200, etc.).

Supports multiple concurrent sweeps via --sweep (repeatable) and/or an
--active-file manifest (re-read each cycle, so sweeps can be added mid-run).

Usage:
    # Single sweep (backwards compatible):
    python sweep_autoscaler.py \\
        --sweep sweeps/bpe16384-n3-n5.yaml \\
        --profile "dw87long|4|6-00:00:00|--qos=dw87long" \\
        --interval 120

    # Multi-sweep via manifest:
    python sweep_autoscaler.py \\
        --active-file sweeps/.active \\
        --profile "dw87long|4|6-00:00:00|--qos=dw87long" \\
        --interval 120

    # Combined (CLI sweeps + manifest):
    python sweep_autoscaler.py \\
        --sweep sweeps/base.yaml \\
        --active-file sweeps/.active \\
        --profile "dw87long|4|6-00:00:00|--qos=dw87long"

    # Dry run (prints sbatch commands without submitting):
    python sweep_autoscaler.py --sweep ... --profile ... --dry-run --once
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from sweep_runner import (
    STORAGE_ROOT,
    build_config,
    check_local_checkpoint,
    deterministic_out_dir,
    deterministic_run_id,
    expand_grid,
    override_summary,
    parse_sweep_yaml,
)
from src.experiment import slug


# ---------------------------------------------------------------------------
# GPU tier hierarchy for rescheduling
# ---------------------------------------------------------------------------

GPU_TIER = {"B200": 4, "H200": 3, "H100": 2, "A100": 1}
MIN_REMAINING_TIME_SECS = 3600   # 1 hour minimum on target job
MAX_PROGRESS_RATIO = 0.8         # don't preempt runs >80% done


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[autoscaler {ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Profile configuration
# ---------------------------------------------------------------------------

@dataclass
class PartitionProfile:
    name: str           # e.g. "dw87long", "cs-b200"
    max_jobs: int       # max concurrent Slurm jobs for this profile
    time_limit: str     # e.g. "6-00:00:00"
    sbatch_flags: str   # e.g. "--qos=cs --gres=gpu:b200:8"
    gpus_per_job: int   # GPUs per Slurm job (default 8)


def parse_profile(spec: str, gpus_per_job: int) -> PartitionProfile:
    """Parse 'name|max_jobs|time_limit|sbatch_flags' into PartitionProfile."""
    parts = spec.split("|", maxsplit=3)
    if len(parts) < 4:
        raise ValueError(
            f"Profile spec must be name|max_jobs|time_limit|sbatch_flags, got: {spec!r}"
        )
    name = parts[0]
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        raise ValueError(f"Profile name must be alphanumeric/hyphens/underscores: {name!r}")
    return PartitionProfile(
        name=name,
        max_jobs=int(parts[1]),
        time_limit=parts[2],
        sbatch_flags=parts[3],
        gpus_per_job=gpus_per_job,
    )


def job_name_for_profile(profile: PartitionProfile) -> str:
    return f"tc-{profile.name}"


# ---------------------------------------------------------------------------
# Sweep state scanning
# ---------------------------------------------------------------------------

@dataclass
class ConfigCandidate:
    run_id: str
    config_label: str
    out_dir: str
    base_toml: str | None
    overrides: dict
    has_checkpoint: bool
    iter_num: int
    max_iters: int
    sweep_path: str = ""


@dataclass
class SweepState:
    total_configs: int
    done_configs: int
    remaining: int
    active_gpus: int  # from heartbeats
    assigned_run_ids: set  # run_ids currently in assignment files
    unassigned_candidates: list  # ConfigCandidate list ready for assignment


def scan_sweep_state(
    sweep_path: str,
    heartbeats: list[dict] | None = None,
    freed_run_ids: set[str] | None = None,
) -> SweepState:
    """Parse sweep grid, scan filesystem for progress, compute assignable candidates."""
    base_toml, project, group, parameters = parse_sweep_yaml(sweep_path)
    grid = expand_grid(parameters)
    project_slug = slug(project)
    freed = freed_run_ids or set()

    # Read existing assignments
    current_assignments = scan_assignments(project_slug)
    assigned_run_ids = {
        asgn["run_id"] for asgn in current_assignments.values()
        if "run_id" in asgn
    }

    # Also treat workers currently training (per heartbeat) as assigned,
    # but only if they have a corresponding assignment file (to avoid
    # re-locking configs reclaimed from stuck/dead workers)
    if heartbeats:
        for hb in heartbeats:
            if hb.get("state") == "training" and hb.get("run_id"):
                rid = hb["run_id"]
                if rid in assigned_run_ids or rid in freed:
                    continue  # already counted or explicitly freed
                # Only trust heartbeat if worker has an assignment file
                wid = hb.get("worker_id", "")
                if wid in current_assignments:
                    assigned_run_ids.add(rid)

    total = len(grid)
    done = 0
    resumable: list[ConfigCandidate] = []
    fresh: list[ConfigCandidate] = []

    for overrides in grid:
        config = build_config(base_toml, overrides)
        run_id = deterministic_run_id(config)
        out_dir = deterministic_out_dir(config, project, group)
        max_iters = int(config.training.max_iters)

        has_ckpt, iter_num = check_local_checkpoint(out_dir)

        # Check if done
        if has_ckpt and iter_num >= max_iters:
            done += 1
            continue
        final_step = os.path.join(out_dir, "checkpoints", f"step-{max_iters:06d}")
        if os.path.isdir(final_step):
            done += 1
            continue

        # Skip if already assigned or being trained
        if run_id in assigned_run_ids:
            continue

        candidate = ConfigCandidate(
            run_id=run_id,
            config_label=override_summary(overrides),
            out_dir=out_dir,
            base_toml=base_toml,
            overrides=overrides,
            has_checkpoint=has_ckpt,
            iter_num=iter_num,
            max_iters=max_iters,
            sweep_path=sweep_path,
        )
        if has_ckpt:
            resumable.append(candidate)
        else:
            fresh.append(candidate)

    # Deterministic ordering within each category
    resumable.sort(key=lambda c: c.run_id)
    fresh.sort(key=lambda c: c.run_id)

    if heartbeats is None:
        heartbeats = read_heartbeats(project_slug)
    active_gpus = count_active_gpus(heartbeats)

    return SweepState(
        total_configs=total,
        done_configs=done,
        remaining=total - done,
        active_gpus=active_gpus,
        assigned_run_ids=assigned_run_ids,
        unassigned_candidates=resumable + fresh,
    )


# ---------------------------------------------------------------------------
# Multi-sweep support
# ---------------------------------------------------------------------------

def read_active_sweeps(active_file: str) -> list[str]:
    """Read manifest file listing sweep YAML paths (one per line).

    Skips blank lines and lines starting with '#'. Returns empty list if
    the file is missing (fail-open).
    """
    try:
        with open(active_file) as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return []
    paths = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            paths.append(stripped)
    return paths


def validate_sweep_project(sweep_paths: list[str]) -> str:
    """Validate all sweeps share the same project and return the project_slug.

    This is a lightweight check (parses YAML headers only, no filesystem scan).
    """
    if not sweep_paths:
        raise ValueError("No sweep paths provided")

    project_slugs = {}
    for sp in sweep_paths:
        _, project, _, _ = parse_sweep_yaml(sp)
        ps = slug(project)
        project_slugs[sp] = ps

    unique_slugs = set(project_slugs.values())
    if len(unique_slugs) > 1:
        details = ", ".join(f"{sp} -> {ps}" for sp, ps in project_slugs.items())
        raise ValueError(f"All sweeps must share the same project, got: {details}")

    return next(iter(unique_slugs))


def scan_all_sweeps(
    sweep_paths: list[str],
    heartbeats: list[dict] | None = None,
    freed_run_ids: set[str] | None = None,
) -> tuple[str, SweepState]:
    """Scan multiple sweeps and merge into a single SweepState.

    All sweeps must share the same project (validated here).
    Candidates are deduplicated by run_id — first sweep wins.

    Returns (project_slug, merged_state).
    """
    project_slug = validate_sweep_project(sweep_paths)

    # Scan each sweep and merge
    total = 0
    done = 0
    seen_run_ids: set[str] = set()
    merged_candidates: list[ConfigCandidate] = []
    merged_assigned: set[str] = set()
    active_gpus = 0

    for sp in sweep_paths:
        state = scan_sweep_state(sp, heartbeats=heartbeats, freed_run_ids=freed_run_ids)
        total += state.total_configs
        done += state.done_configs
        merged_assigned |= state.assigned_run_ids
        active_gpus = state.active_gpus  # same heartbeats, same count

        for candidate in state.unassigned_candidates:
            if candidate.run_id not in seen_run_ids:
                seen_run_ids.add(candidate.run_id)
                merged_candidates.append(candidate)

    return project_slug, SweepState(
        total_configs=total,
        done_configs=done,
        remaining=total - done,
        active_gpus=active_gpus,
        assigned_run_ids=merged_assigned,
        unassigned_candidates=merged_candidates,
    )


# ---------------------------------------------------------------------------
# Heartbeat reading
# ---------------------------------------------------------------------------

def read_heartbeats(project_slug: str) -> list[dict]:
    """Read heartbeat JSON files from .heartbeats/ directory."""
    hb_dir = Path(STORAGE_ROOT) / "out" / project_slug / ".heartbeats"
    if not hb_dir.is_dir():
        return []
    heartbeats = []
    now = time.time()
    for f in hb_dir.iterdir():
        if not f.name.endswith(".json") or ".tmp." in f.name:
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            # Only include fresh heartbeats
            if now - data.get("timestamp", 0) < 300:
                heartbeats.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return heartbeats


def count_active_gpus(heartbeats: list[dict]) -> int:
    """Count heartbeats where state=='training'."""
    return sum(1 for hb in heartbeats if hb.get("state") == "training")


# ---------------------------------------------------------------------------
# Assignment management (centralized work assignment)
# ---------------------------------------------------------------------------

def scan_assignments(project_slug: str) -> dict[str, dict]:
    """Read all assignment files. Returns {worker_id: assignment_dict}."""
    assign_dir = Path(STORAGE_ROOT) / "out" / project_slug / ".assignments"
    if not assign_dir.is_dir():
        return {}
    assignments = {}
    for f in assign_dir.iterdir():
        if not f.name.endswith(".json") or ".tmp." in f.name:
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            # Derive worker_id from filename (reverse of replace(":", "_"))
            wid = data.get("worker_id", f.stem.replace("_", ":", 1))
            assignments[wid] = data
        except (json.JSONDecodeError, OSError):
            continue
    return assignments


def write_assignment(project_slug: str, worker_id: str, assignment: dict):
    """Atomically write an assignment file for a worker."""
    assign_dir = Path(STORAGE_ROOT) / "out" / project_slug / ".assignments"
    assign_dir.mkdir(parents=True, exist_ok=True)

    safe_name = worker_id.replace(":", "_")
    target = assign_dir / f"{safe_name}.json"
    tmp = assign_dir / f"{safe_name}.json.tmp.{os.getpid()}"

    assignment["worker_id"] = worker_id
    with open(tmp, "w") as f:
        json.dump(assignment, f, indent=2)
    os.replace(str(tmp), str(target))


def cleanup_stale_assignments(
    project_slug: str,
    heartbeats: list[dict],
) -> list[str]:
    """Remove assignments for dead workers. Returns list of freed run_ids."""
    fresh_worker_ids = {hb["worker_id"] for hb in heartbeats}
    assignments = scan_assignments(project_slug)
    freed = []

    assign_dir = Path(STORAGE_ROOT) / "out" / project_slug / ".assignments"

    for wid, asgn in assignments.items():
        if asgn.get("action") == "shutdown":
            # Shutdown signals for dead workers can also be cleaned up
            if wid not in fresh_worker_ids:
                safe_name = wid.replace(":", "_")
                try:
                    os.remove(assign_dir / f"{safe_name}.json")
                except OSError:
                    pass
            continue

        if wid not in fresh_worker_ids:
            # Worker is dead — reclaim this assignment
            safe_name = wid.replace(":", "_")
            try:
                os.remove(assign_dir / f"{safe_name}.json")
                freed_rid = asgn.get("run_id", "unknown")
                log(f"  Reclaimed {freed_rid} from dead worker {wid}")
                freed.append(freed_rid)
            except OSError:
                pass
        elif asgn.get("run_id"):
            hb = next((h for h in heartbeats if h.get("worker_id") == wid), None)
            if not hb:
                continue
            age = time.time() - asgn.get("assigned_at", 0)

            # Worker is alive but idle — assignment should have been deleted
            if hb.get("state") == "idle" and age > 600:  # 10 minutes
                safe_name = wid.replace(":", "_")
                try:
                    os.remove(assign_dir / f"{safe_name}.json")
                    freed_rid = asgn.get("run_id", "unknown")
                    log(f"  Reclaimed {freed_rid} from idle worker {wid} (stale assignment, {age/60:.0f}m)")
                    freed.append(freed_rid)
                except OSError:
                    pass

            # Worker is "training" but stuck at iter 0 for >60 min (e.g. torch.compile hang)
            elif hb.get("state") == "training" and hb.get("iter_num", 0) == 0 and age > 3600:
                safe_name = wid.replace(":", "_")
                try:
                    os.remove(assign_dir / f"{safe_name}.json")
                    freed_rid = asgn.get("run_id", "unknown")
                    log(f"  Reclaimed {freed_rid} from stuck worker {wid} (iter_num=0 for {age/60:.0f}m)")
                    freed.append(freed_rid)
                except OSError:
                    pass

    return freed


# ---------------------------------------------------------------------------
# GPU rescheduling: migrate runs from worse GPUs to idle better GPUs
# ---------------------------------------------------------------------------

def get_job_remaining_time(job_id: str) -> int | None:
    """Get remaining wall time for a Slurm job in seconds.

    Returns None on failure (fail-open: skip rescheduling for this worker).
    """
    try:
        result = subprocess.run(
            ["squeue", "-j", job_id, "-o", "%L", "--noheader"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return None
        time_str = result.stdout.strip()
        if not time_str or time_str == "INVALID":
            return None
        # Parse time formats: "5-23:45:00", "23:45:00", "45:00", "00"
        parts = time_str.split("-")
        days = 0
        if len(parts) == 2:
            days = int(parts[0])
            hms = parts[1]
        else:
            hms = parts[0]
        hms_parts = hms.split(":")
        if len(hms_parts) == 3:
            h, m, s = int(hms_parts[0]), int(hms_parts[1]), int(hms_parts[2])
        elif len(hms_parts) == 2:
            h, m, s = 0, int(hms_parts[0]), int(hms_parts[1])
        else:
            h, m, s = 0, 0, int(hms_parts[0])
        return days * 86400 + h * 3600 + m * 60 + s
    except Exception:
        return None


def reschedule_for_better_gpus(
    project_slug: str,
    heartbeats: list[dict],
    min_remaining_time: int = MIN_REMAINING_TIME_SECS,
    max_progress_ratio: float = MAX_PROGRESS_RATIO,
    dry_run: bool = False,
) -> int:
    """Detect GPU tier mismatches and write preempt signals to migrate runs.

    Returns the number of preemption signals written (max 1 per cycle).
    """
    current_assignments = scan_assignments(project_slug)
    assigned_worker_ids = set(current_assignments.keys())

    # Partition heartbeats into idle (unassigned) and training workers
    idle_by_tier: list[dict] = []
    training_by_tier: list[dict] = []

    for hb in heartbeats:
        gpu_type = hb.get("gpu_type", "unknown")
        tier = GPU_TIER.get(gpu_type)
        if tier is None:
            continue  # skip unknown GPU types

        wid = hb.get("worker_id", "")
        if hb.get("state") == "idle" and wid not in assigned_worker_ids:
            idle_by_tier.append({**hb, "_tier": tier})
        elif hb.get("state") == "training" and hb.get("run_id"):
            training_by_tier.append({**hb, "_tier": tier})

    if not idle_by_tier or not training_by_tier:
        return 0

    # Sort idle workers by tier descending (best GPUs first)
    idle_by_tier.sort(key=lambda h: h["_tier"], reverse=True)
    # Sort training workers by tier ascending (worst GPUs first — best preempt targets)
    training_by_tier.sort(key=lambda h: h["_tier"])

    preempt_dir = Path(STORAGE_ROOT) / "out" / project_slug / ".preempt"

    preemptions = 0
    used_training_workers = set()

    for idle_hb in idle_by_tier:
        if preemptions >= 1:  # max 1 preemption per cycle
            break

        idle_tier = idle_hb["_tier"]
        idle_job_id = idle_hb.get("slurm_job_id")

        # Check remaining time on the idle worker's Slurm job
        if idle_job_id:
            remaining = get_job_remaining_time(idle_job_id)
            if remaining is not None and remaining < min_remaining_time:
                log(f"  Reschedule skip: {idle_hb['worker_id']} has only {remaining}s remaining (need {min_remaining_time}s)")
                continue

        # Find a training worker on a strictly worse GPU
        for train_hb in training_by_tier:
            train_wid = train_hb.get("worker_id", "")
            if train_wid in used_training_workers:
                continue
            train_tier = train_hb["_tier"]
            if train_tier >= idle_tier:
                break  # sorted ascending; no worse GPUs left

            # Check progress — skip if run is nearly done
            max_iters = train_hb.get("max_iters", 0)
            iter_num = train_hb.get("iter_num", 0)
            if max_iters > 0 and iter_num / max_iters > max_progress_ratio:
                log(f"  Reschedule skip: {train_hb.get('run_id')} is {iter_num}/{max_iters} ({iter_num/max_iters:.0%} done)")
                continue

            # Write preemption signal
            safe_name = train_wid.replace(":", "_")
            signal_data = {
                "reason": "gpu_upgrade",
                "source_worker": train_wid,
                "source_gpu": train_hb.get("gpu_type", "unknown"),
                "target_worker": idle_hb.get("worker_id", ""),
                "target_gpu": idle_hb.get("gpu_type", "unknown"),
                "run_id": train_hb.get("run_id"),
                "timestamp": time.time(),
            }

            if dry_run:
                log(f"  [DRY RUN] Would preempt {train_hb.get('run_id')} on {train_wid} "
                    f"({train_hb.get('gpu_type')}) -> {idle_hb.get('worker_id')} "
                    f"({idle_hb.get('gpu_type')})")
            else:
                preempt_dir.mkdir(parents=True, exist_ok=True)
                signal_path = preempt_dir / f"{safe_name}.json"
                tmp_path = preempt_dir / f"{safe_name}.json.tmp.{os.getpid()}"
                with open(tmp_path, "w") as f:
                    json.dump(signal_data, f, indent=2)
                os.replace(str(tmp_path), str(signal_path))
                log(f"  Preempt: {train_hb.get('run_id')} on {train_wid} "
                    f"({train_hb.get('gpu_type')}) -> {idle_hb.get('worker_id')} "
                    f"({idle_hb.get('gpu_type')}) [iter {iter_num}/{max_iters}]")

            used_training_workers.add(train_wid)
            preemptions += 1
            break

    # Cleanup stale preempt signal files (older than 5 minutes)
    if preempt_dir.is_dir():
        now = time.time()
        for f in preempt_dir.iterdir():
            if not f.name.endswith(".json") or ".tmp." in f.name:
                continue
            try:
                if now - f.stat().st_mtime > 300:
                    f.unlink()
                    log(f"  Cleaned up stale preempt signal: {f.name}")
            except OSError:
                pass

    return preemptions


def assign_work_to_idle_workers(
    project_slug: str,
    heartbeats: list[dict],
    sweep_state: SweepState,
):
    """Assign configs to idle workers."""
    # Find idle workers that don't already have an assignment file
    current_assignments = scan_assignments(project_slug)
    assigned_worker_ids = set(current_assignments.keys())

    idle_hbs = [
        hb for hb in heartbeats
        if hb.get("state") == "idle" and hb["worker_id"] not in assigned_worker_ids
    ]
    # Sort by GPU tier descending so best GPUs get work first
    idle_hbs.sort(
        key=lambda hb: GPU_TIER.get(hb.get("gpu_type", "unknown"), 0),
        reverse=True,
    )
    idle_workers = [hb["worker_id"] for hb in idle_hbs]

    if not idle_workers:
        return

    candidates = list(sweep_state.unassigned_candidates)

    if not candidates:
        if sweep_state.remaining == 0:
            # All work done — send shutdown to idle workers
            for wid in idle_workers:
                write_assignment(project_slug, wid, {"action": "shutdown"})
                log(f"  Sent shutdown to {wid}")
        return

    # Assign one config per idle worker
    for wid in idle_workers:
        if not candidates:
            break
        candidate = candidates.pop(0)
        assignment = {
            "run_id": candidate.run_id,
            "config_label": candidate.config_label,
            "out_dir": candidate.out_dir,
            "base_toml": candidate.base_toml,
            "overrides": candidate.overrides,
            "assigned_at": time.time(),
            "sweep_path": candidate.sweep_path,
        }
        write_assignment(project_slug, wid, assignment)
        log(f"  Assigned {candidate.run_id} -> {wid}")


# ---------------------------------------------------------------------------
# Slurm job querying
# ---------------------------------------------------------------------------

def get_sweep_jobs() -> list[dict] | None:
    """Query squeue for all user's jobs. Returns None if squeue fails."""
    try:
        result = subprocess.run(
            ["squeue", "--me", "--noheader", "-o", "%i|%T|%j"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            log(f"WARNING: squeue failed (rc={result.returncode})")
            return None
    except subprocess.TimeoutExpired:
        log("WARNING: squeue timed out")
        return None
    except FileNotFoundError:
        log("WARNING: squeue not found")
        return None

    jobs = []
    for line in result.stdout.strip().splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue
        jobs.append({
            "jobid": parts[0].strip(),
            "state": parts[1].strip(),
            "name": parts[2].strip(),
        })
    return jobs


def count_jobs_per_profile(
    jobs: list[dict],
    profiles: list[PartitionProfile],
) -> dict[str, dict[str, int]]:
    """Match squeue jobs to profiles by job name, count running/pending."""
    profile_names = {job_name_for_profile(p): p.name for p in profiles}
    counts: dict[str, dict[str, int]] = {
        p.name: {"running": 0, "pending": 0} for p in profiles
    }

    for job in jobs:
        pname = profile_names.get(job["name"])
        if pname is None:
            continue
        if job["state"] == "RUNNING":
            counts[pname]["running"] += 1
        elif job["state"] == "PENDING":
            counts[pname]["pending"] += 1

    return counts


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def decide_submissions(
    sweep_state: SweepState,
    profiles: list[PartitionProfile],
    jobs_per_profile: dict[str, dict[str, int]],
) -> dict[str, int]:
    """For each profile, decide how many jobs to submit.

    Returns {profile_name: num_jobs_to_submit}.
    """
    remaining = sweep_state.remaining

    if remaining <= 0:
        return {p.name: 0 for p in profiles}

    # Total GPUs already provisioned (running + pending) across all profiles
    total_provisioned_gpus = sum(
        (info["running"] + info["pending"]) * p.gpus_per_job
        for p, info in ((p, jobs_per_profile.get(p.name, {"running": 0, "pending": 0}))
                        for p in profiles)
    )

    result: dict[str, int] = {}
    for profile in profiles:
        info = jobs_per_profile.get(profile.name, {"running": 0, "pending": 0})
        current = info["running"] + info["pending"]
        headroom = profile.max_jobs - current

        if headroom <= 0:
            result[profile.name] = 0
            continue

        # Don't over-provision: only submit if remaining work exceeds current GPUs
        if total_provisioned_gpus >= remaining:
            result[profile.name] = 0
            continue

        # Conservative: max 1 job per profile per cycle
        result[profile.name] = 1
        total_provisioned_gpus += profile.gpus_per_job

    return result


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------

def submit_job(
    profile: PartitionProfile,
    sweep_path: str,
    proxykit_dir: str,
    dry_run: bool,
) -> str | None:
    """Submit a Slurm job for the given profile. Returns job ID or None."""
    cmd = [
        "sbatch",
        f"--job-name={job_name_for_profile(profile)}",
        f"--time={profile.time_limit}",
        "--mem=0",
        *shlex.split(profile.sbatch_flags),
        "./slurm/run_sweep_runner.sbatch",
        sweep_path,
    ]

    env = os.environ.copy()
    env["PROXYKIT_DIR"] = proxykit_dir

    if dry_run:
        log(f"  [DRY RUN] PROXYKIT_DIR={proxykit_dir} {' '.join(cmd)}")
        return None

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=30,
        )
    except subprocess.TimeoutExpired:
        log(f"  ERROR: sbatch timed out for profile {profile.name}")
        return None

    if result.returncode != 0:
        log(f"  ERROR: sbatch failed for {profile.name}: {result.stderr.strip()}")
        return None

    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        return match.group(1)
    log(f"  WARNING: Could not parse sbatch output: {result.stdout.strip()}")
    return None


# ---------------------------------------------------------------------------
# Scale-down
# ---------------------------------------------------------------------------

def _cancel_pending_jobs(
    jobs: list[dict],
    profiles: list[PartitionProfile],
    dry_run: bool,
):
    """Cancel pending (queued but not running) jobs managed by this autoscaler."""
    managed_names = {job_name_for_profile(p) for p in profiles}
    pending_ids = [
        j["jobid"] for j in jobs
        if j["name"] in managed_names and j["state"] == "PENDING"
    ]
    if not pending_ids:
        return
    log(f"  Cancelling {len(pending_ids)} pending job(s): {', '.join(pending_ids)}")
    if dry_run:
        log(f"  [DRY RUN] Would run: scancel {' '.join(pending_ids)}")
        return
    try:
        subprocess.run(
            ["scancel"] + pending_ids,
            capture_output=True, text=True, timeout=15,
        )
    except Exception as e:
        log(f"  WARNING: scancel failed: {e}")


# ---------------------------------------------------------------------------
# Cycle
# ---------------------------------------------------------------------------

def cycle(
    sweep_paths: list[str],
    profiles: list[PartitionProfile],
    proxykit_dir: str,
    dry_run: bool,
    reschedule: bool = False,
    min_remaining_time: int = MIN_REMAINING_TIME_SECS,
):
    """One scan-decide-submit cycle."""
    log(f"=== Scan cycle ({len(sweep_paths)} active sweep(s)) ===")

    # Derive project_slug from all sweeps (validated to be the same)
    project_slug = validate_sweep_project(sweep_paths)

    # 1a. Read heartbeats
    heartbeats = read_heartbeats(project_slug)

    # 1b. Cleanup stale assignments (dead workers)
    freed = cleanup_stale_assignments(project_slug, heartbeats)
    if freed:
        log(f"  Freed {len(freed)} assignment(s) from dead workers")

    # 1c. Reschedule: migrate runs from worse GPUs to idle better GPUs
    if reschedule:
        n_preempted = reschedule_for_better_gpus(
            project_slug, heartbeats,
            min_remaining_time=min_remaining_time,
            dry_run=dry_run,
        )
        if n_preempted:
            log(f"  Triggered {n_preempted} GPU migration(s)")

    # 1d. Scan all sweeps and merge state
    project_slug, sweep_state = scan_all_sweeps(
        sweep_paths, heartbeats=heartbeats, freed_run_ids=set(freed),
    )
    n_assigned = len(sweep_state.assigned_run_ids)
    n_assignable = len(sweep_state.unassigned_candidates)
    log(f"Grid: {sweep_state.total_configs} total, "
        f"{sweep_state.done_configs} done, "
        f"{sweep_state.remaining} remaining, "
        f"{n_assigned} assigned, "
        f"{n_assignable} assignable, "
        f"{sweep_state.active_gpus} active GPUs (heartbeats)")

    # 1e. Assign work to idle workers
    assign_work_to_idle_workers(project_slug, heartbeats, sweep_state)

    # 2. Query squeue
    jobs = get_sweep_jobs()
    if jobs is None:
        log("Skipping cycle (squeue unavailable)")
        return

    jobs_per_profile = count_jobs_per_profile(jobs, profiles)

    # Also count unmanaged jobs (manually submitted)
    managed_names = {job_name_for_profile(p) for p in profiles}
    unmanaged = sum(
        1 for j in jobs
        if j["name"] not in managed_names and "translation-compression" in j["name"]
    )

    for profile in profiles:
        info = jobs_per_profile[profile.name]
        log(f"  {profile.name}: {info['running']}R {info['pending']}PD "
            f"(max {profile.max_jobs})")
    if unmanaged:
        log(f"  + {unmanaged} unmanaged job(s)")

    # 3. Decide
    submissions = decide_submissions(sweep_state, profiles, jobs_per_profile)

    # 4. Submit (use first sweep for sbatch — worker only needs project_slug)
    any_submitted = False
    submit_sweep = sweep_paths[0]
    for profile in profiles:
        n = submissions.get(profile.name, 0)
        for _ in range(n):
            job_id = submit_job(profile, submit_sweep, proxykit_dir, dry_run)
            if job_id:
                log(f"  Submitted job {job_id} for {profile.name}")
                any_submitted = True
            elif dry_run:
                any_submitted = True  # dry run counts as "would submit"

    if not any_submitted:
        if sweep_state.remaining == 0:
            log("All configs done. Nothing to submit.")
            # Cancel pending jobs (no point queueing for finished work)
            _cancel_pending_jobs(jobs, profiles, dry_run)
        else:
            log("No new jobs needed (existing jobs sufficient).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Autoscaler: maintain target GPU count by submitting Slurm jobs"
    )
    parser.add_argument("--sweep", action="append", dest="sweeps", default=None,
                        help="Path to sweep YAML (repeatable)")
    parser.add_argument("--active-file", default=None,
                        help="Path to manifest file listing sweep YAMLs (e.g. sweeps/.active)")
    parser.add_argument("--profile", action="append", dest="profiles", required=True,
                        metavar="NAME|MAX_JOBS|TIME|SBATCH_FLAGS",
                        help="Partition profile (repeatable). Example: 'dw87long|4|6-00:00:00|--qos=dw87long'")
    parser.add_argument("--interval", type=int, default=120,
                        help="Seconds between scan cycles (default: 120)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sbatch commands without submitting")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle and exit")
    parser.add_argument("--gpus-per-job", type=int, default=8,
                        help="GPUs per Slurm job (default: 8)")
    parser.add_argument("--proxykit-dir", default=None,
                        help="Override PROXYKIT_DIR (default: $PWD/slurm)")
    parser.add_argument("--reschedule", action="store_true",
                        help="Enable GPU rescheduling: migrate runs from worse GPUs to idle better GPUs")
    parser.add_argument("--min-remaining-time", type=int, default=MIN_REMAINING_TIME_SECS,
                        help=f"Min remaining Slurm time (seconds) on target job for rescheduling (default: {MIN_REMAINING_TIME_SECS})")
    args = parser.parse_args()

    if not args.sweeps and not args.active_file:
        parser.error("At least one of --sweep or --active-file is required")

    # Parse profiles
    profiles = []
    seen_names: set[str] = set()
    for spec in args.profiles:
        p = parse_profile(spec, args.gpus_per_job)
        if p.name in seen_names:
            print(f"ERROR: Duplicate profile name: {p.name}", file=sys.stderr)
            sys.exit(1)
        seen_names.add(p.name)
        profiles.append(p)

    proxykit_dir = args.proxykit_dir or os.path.join(os.getcwd(), "slurm")
    if not os.path.isdir(proxykit_dir):
        print(f"ERROR: PROXYKIT_DIR not found: {proxykit_dir}", file=sys.stderr)
        sys.exit(1)

    cli_sweeps = args.sweeps or []
    if cli_sweeps:
        log(f"CLI sweeps: {', '.join(cli_sweeps)}")
    if args.active_file:
        log(f"Active file: {args.active_file}")
    log(f"Profiles: {', '.join(f'{p.name} (max {p.max_jobs})' for p in profiles)}")
    log(f"Interval: {args.interval}s | Dry run: {args.dry_run} | Reschedule: {args.reschedule}")
    log("")

    while True:
        try:
            # Build sweep list each cycle (active file is re-read)
            sweep_paths = list(cli_sweeps)
            if args.active_file:
                from_file = read_active_sweeps(args.active_file)
                sweep_paths.extend(p for p in from_file if p not in sweep_paths)

            if not sweep_paths:
                log("WARNING: No sweep paths found (--sweep and active file both empty). Skipping cycle.")
            else:
                log(f"{len(sweep_paths)} active sweep(s): {', '.join(sweep_paths)}")
                cycle(sweep_paths, profiles, proxykit_dir, args.dry_run,
                      reschedule=args.reschedule,
                      min_remaining_time=args.min_remaining_time)
        except KeyboardInterrupt:
            log("Interrupted. Exiting.")
            break
        except Exception as e:
            log(f"ERROR in cycle: {e}")
            import traceback
            traceback.print_exc()

        if args.once:
            break

        log(f"Sleeping {args.interval}s...")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            log("Interrupted. Exiting.")
            break


if __name__ == "__main__":
    main()
