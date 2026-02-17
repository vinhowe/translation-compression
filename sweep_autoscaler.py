"""
Lightweight autoscaler for sweep_runner Slurm jobs.

Periodically scans sweep progress and submits Slurm jobs to maintain target
GPU counts across multiple partition profiles (H100, B200, etc.).

Usage:
    python sweep_autoscaler.py \\
        --sweep sweeps/bpe16384-n3-n5.yaml \\
        --profile "dw87long|4|6-00:00:00|--qos=dw87long" \\
        --profile "cs-b200|2|1-00:00:00|--qos=cs --gres=gpu:b200:8" \\
        --interval 120

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
    parse_sweep_yaml,
)
from src.experiment import slug


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
class SweepState:
    total_configs: int
    done_configs: int
    remaining: int
    active_gpus: int  # from heartbeats


def scan_sweep_state(sweep_path: str) -> SweepState:
    """Parse sweep grid, scan filesystem for progress, read heartbeats."""
    base_toml, project, group, parameters = parse_sweep_yaml(sweep_path)
    grid = expand_grid(parameters)
    project_slug = slug(project)

    total = len(grid)
    done = 0

    for overrides in grid:
        config = build_config(base_toml, overrides)
        out_dir = deterministic_out_dir(config, project, group)
        max_iters = int(config.training.max_iters)

        has_ckpt, iter_num = check_local_checkpoint(out_dir)
        if has_ckpt and iter_num >= max_iters:
            done += 1
            continue

        # Check for final step dir (stale latest symlink)
        final_step = os.path.join(out_dir, "checkpoints", f"step-{max_iters:06d}")
        if os.path.isdir(final_step):
            done += 1

    # Count active GPUs from heartbeats
    active_gpus = count_active_gpus(read_heartbeats(project_slug))

    return SweepState(
        total_configs=total,
        done_configs=done,
        remaining=total - done,
        active_gpus=active_gpus,
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
    sweep_path: str,
    profiles: list[PartitionProfile],
    proxykit_dir: str,
    dry_run: bool,
):
    """One scan-decide-submit cycle."""
    log("=== Scan cycle ===")

    # 1. Scan sweep state
    sweep_state = scan_sweep_state(sweep_path)
    log(f"Grid: {sweep_state.total_configs} total, "
        f"{sweep_state.done_configs} done, "
        f"{sweep_state.remaining} remaining, "
        f"{sweep_state.active_gpus} active GPUs (heartbeats)")

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

    # 4. Submit
    any_submitted = False
    for profile in profiles:
        n = submissions.get(profile.name, 0)
        for _ in range(n):
            job_id = submit_job(profile, sweep_path, proxykit_dir, dry_run)
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
    parser.add_argument("--sweep", required=True,
                        help="Path to sweep YAML")
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
    args = parser.parse_args()

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

    log(f"Sweep: {args.sweep}")
    log(f"Profiles: {', '.join(f'{p.name} (max {p.max_jobs})' for p in profiles)}")
    log(f"Interval: {args.interval}s | Dry run: {args.dry_run}")
    log("")

    while True:
        try:
            cycle(args.sweep, profiles, proxykit_dir, args.dry_run)
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
