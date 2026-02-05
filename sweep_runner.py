"""
Sweep runner that replaces `wandb agent` for scheduling grid sweeps across
two clusters (Slurm + non-Slurm).

Uses file locks for intra-cluster coordination and wandb run registry for
cross-cluster visibility.

Usage:
    # Run the sweep (one process per GPU)
    python sweep_runner.py --sweep sweeps/my-grid.yaml --cluster slurm [--wandb-buffer]

    # Show status table
    python sweep_runner.py --sweep sweeps/my-grid.yaml --status
"""

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, fields, replace
from typing import Any

import yaml

from src.config.job_config import JobConfig
from src.config.manager import ConfigManager
from src.config.presets import apply_bpe16384_batch_config, apply_size_tier
from src.experiment import cfg_hash, slug

STORAGE_ROOT = os.environ.get(
    "TC_STORAGE_ROOT", "/mnt/pccfs2/backed_up/vin/dev/translation-compression"
)


def check_git_is_current() -> tuple[bool, str]:
    """Check that local branch is not behind its remote tracking branch.

    Returns (ok, message). ok is True if up-to-date or ahead.
    """
    try:
        # Fetch latest remote refs (lightweight, no merge)
        subprocess.run(
            ["git", "fetch", "--quiet"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        # Compare local HEAD to upstream
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD..@{upstream}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            # No upstream configured — can't check, allow it
            return True, "no upstream tracking branch configured"
        behind = int(result.stdout.strip())
        if behind > 0:
            return False, f"local branch is {behind} commit(s) behind remote"
        return True, "up to date"
    except subprocess.TimeoutExpired:
        return True, "git fetch timed out, skipping check"
    except Exception as e:
        return True, f"could not check git status: {e}"


def parse_sweep_yaml(path: str) -> tuple[str | None, str, str | None, dict]:
    """Parse sweep YAML. Returns (base_toml_path, wandb_project, group_override, parameters_dict).

    The base TOML path is extracted from the command section (look for .toml in command list).
    If no command section exists, base_toml_path is None.
    """
    with open(path) as f:
        sweep = yaml.safe_load(f)

    project = sweep.get("project", "translation-compression")
    parameters = sweep.get("parameters", {})
    group = sweep.get("name", None)

    # Extract base TOML path from command section
    base_toml: str | None = None
    command = sweep.get("command", [])
    for i, item in enumerate(command):
        if isinstance(item, str) and item.endswith(".toml"):
            base_toml = item
            break
        # Also check if a .toml path follows --job.config-file
        if isinstance(item, str) and item in ("--job.config-file", "--job.config_file"):
            if i + 1 < len(command):
                candidate = command[i + 1]
                if isinstance(candidate, str) and candidate.endswith(".toml"):
                    base_toml = candidate
                    break

    return base_toml, project, group, parameters


def expand_grid(parameters: dict) -> list[dict]:
    """Expand parameter grid into list of flat override dicts.

    e.g., {"model.size_tier": {"values": ["8-128"]}, "training.seed": {"values": [64, 65]}}
    -> [{"model.size_tier": "8-128", "training.seed": 64},
        {"model.size_tier": "8-128", "training.seed": 65}]
    """
    keys = list(parameters.keys())
    value_lists = []
    for k in keys:
        spec = parameters[k]
        if isinstance(spec, dict) and "values" in spec:
            value_lists.append(spec["values"])
        elif isinstance(spec, dict) and "value" in spec:
            value_lists.append([spec["value"]])
        else:
            value_lists.append([spec])

    configs = []
    for combo in itertools.product(*value_lists):
        configs.append(dict(zip(keys, combo)))
    return configs


def _set_nested(obj: Any, dotted_key: str, value: Any) -> Any:
    """Set a dotted key (e.g., 'model.size_tier') on a frozen dataclass, returning a new copy."""
    parts = dotted_key.split(".")
    if len(parts) == 1:
        return replace(obj, **{parts[0]: value})

    section_name = parts[0]
    rest = ".".join(parts[1:])
    section = getattr(obj, section_name)
    new_section = _set_nested(section, rest, value)
    return replace(obj, **{section_name: new_section})


def build_config(base_toml: str | None, overrides: dict) -> JobConfig:
    """Load base TOML, apply dot-notation overrides, apply presets.

    1. Load TOML via ConfigManager.load_from_toml_file()
    2. Apply overrides by replacing nested fields (split on '.')
    3. Apply apply_size_tier() and apply_bpe16384_batch_config()
    4. Mirror seeds (assignment_seed, uniform_seed) same as train.py
    Returns the final JobConfig.
    """
    cm = ConfigManager()
    if base_toml:
        config = cm.load_from_toml_file(base_toml)
    else:
        config = JobConfig()

    # Apply overrides
    for dotted_key, value in overrides.items():
        # Convert numeric strings if needed
        config = _set_nested(config, dotted_key, value)

    # Apply presets (same order as train.py main())
    config = apply_size_tier(config)
    config = apply_bpe16384_batch_config(config)
    config = replace(
        config,
        experiment=replace(config.experiment, assignment_seed=config.training.seed),
    )
    if config.data.uniform_seed == 0:
        config = replace(
            config,
            data=replace(config.data, uniform_seed=config.training.seed),
        )

    return config


def deterministic_run_id(config: JobConfig) -> str:
    """Return cfg_hash(config) as the run_id."""
    return cfg_hash(config)


def deterministic_out_dir(config: JobConfig, project: str, group: str | None) -> str:
    """Return out/{project}/{group}/{cfg_hash}_s{seed}/"""
    out_root = os.path.join(STORAGE_ROOT, "out")
    project_slug = slug(project)
    group_slug = slug(group or "default")
    h = cfg_hash(config)
    seed = config.training.seed
    return os.path.join(out_root, project_slug, group_slug, f"{h}_s{seed}")


def try_claim(lock_dir: str, run_id: str, cluster: str) -> bool:
    """Atomic file-lock claim. O_CREAT | O_EXCL on {lock_dir}/{run_id}.lock.

    Write cluster name + timestamp to the lock file.
    Returns True if claimed, False if already taken.
    """
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, f"{run_id}.lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        content = json.dumps({
            "cluster": cluster,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid(),
        })
        os.write(fd, content.encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def query_wandb_status(project: str, run_ids: list[str]) -> dict[str, dict]:
    """Query wandb API for run states. Returns {run_id: {"state": ..., "cluster": ...}}.

    States: 'finished', 'running', 'crashed', 'failed', or None (not found).
    One API call to list all runs in the project, filter by our run_ids.
    """
    result: dict[str, dict] = {}
    if not run_ids:
        return result

    try:
        import wandb
        api = wandb.Api()
        run_id_set = set(run_ids)

        # Query all runs in the project; filter client-side by id
        runs = api.runs(project, per_page=1000)
        for run in runs:
            if run.id in run_id_set:
                result[run.id] = {
                    "state": run.state,
                    "cluster": run.config.get("cluster", "unknown"),
                }
    except Exception as e:
        print(f"[sweep] Warning: wandb query failed: {e}")

    return result


def check_local_checkpoint(out_dir: str) -> tuple[bool, int]:
    """Check if out_dir/checkpoints/latest exists.

    Returns (has_checkpoint, iter_num from trainer_state.json).
    """
    latest = os.path.join(out_dir, "checkpoints", "latest")
    if not (os.path.islink(latest) or os.path.isdir(latest)):
        return False, 0

    trainer_state_path = os.path.join(latest, "trainer_state.json")
    if not os.path.exists(trainer_state_path):
        return False, 0

    try:
        with open(trainer_state_path) as f:
            state = json.load(f)
        return True, state.get("iter_num", 0)
    except Exception:
        return False, 0


def run_training(
    config: JobConfig,
    run_id: str,
    out_dir: str,
    cluster: str,
    wandb_buffer: bool,
):
    """Set env vars (RUN_ID, OUT_DIR) and call train.main(config)."""
    os.environ["RUN_ID"] = run_id
    os.environ["OUT_DIR"] = out_dir

    # Apply wandb_buffer and inject cluster tag
    if wandb_buffer:
        config = replace(
            config,
            logging=replace(config.logging, wandb_buffer=True),
        )

    import train
    train.main(config)

    # Clean up env vars
    os.environ.pop("RUN_ID", None)
    os.environ.pop("OUT_DIR", None)


def _override_summary(overrides: dict) -> str:
    """Short string summarizing the override values."""
    parts = []
    for k, v in overrides.items():
        short_key = k.split(".")[-1]
        parts.append(f"{short_key}={v}")
    return " ".join(parts)


def main_loop(args):
    """Main loop: parse sweep, expand grid, claim and run configs."""
    base_toml, project, group, parameters = parse_sweep_yaml(args.sweep)
    grid = expand_grid(parameters)
    print(f"[sweep] Expanded grid: {len(grid)} configs from {args.sweep}")

    while True:
        # Build configs and collect metadata
        candidates = []
        all_run_ids = []
        for overrides in grid:
            config = build_config(base_toml, overrides)
            run_id = deterministic_run_id(config)
            out_dir = deterministic_out_dir(config, project, group)
            all_run_ids.append(run_id)
            candidates.append({
                "config": config,
                "run_id": run_id,
                "out_dir": out_dir,
                "overrides": overrides,
            })

        # Query wandb for all run states (single API call)
        wandb_states = query_wandb_status(project, all_run_ids)

        # Classify candidates
        resumable = []  # Has local checkpoint, not finished
        claimable = []  # Not in wandb, not locked

        lock_dir = os.path.join(STORAGE_ROOT, "out", slug(project), ".locks")

        for c in candidates:
            rid = c["run_id"]
            ws = wandb_states.get(rid, {})
            state = ws.get("state")

            # Skip finished runs
            if state == "finished":
                continue

            has_ckpt, iter_num = check_local_checkpoint(c["out_dir"])
            c["has_ckpt"] = has_ckpt
            c["iter_num"] = iter_num

            if has_ckpt and state != "running":
                # Local partial checkpoint — resume it
                resumable.append(c)
            elif state in ("running",):
                # Currently running somewhere — skip
                continue
            elif state in ("crashed", "failed"):
                # Failed/crashed — reclaimable
                claimable.append(c)
            elif state is None:
                # Not in wandb at all — new work
                lock_path = os.path.join(lock_dir, f"{rid}.lock")
                if os.path.exists(lock_path):
                    continue  # Already claimed by another process
                claimable.append(c)

        # Priority: resume first, then new work
        ordered = resumable + claimable

        if not ordered:
            print("[sweep] No more candidates. Grid exhausted.")
            break

        # Shuffle to reduce collision across parallel GPU workers
        random.shuffle(ordered)
        # But put resumable ones first (they were shuffled among themselves)
        ordered.sort(key=lambda c: not c.get("has_ckpt", False))

        claimed = False
        for c in ordered:
            rid = c["run_id"]
            if c.get("has_ckpt"):
                # Lock file from the original claim already exists — that's
                # expected.  Use a separate resume lock so only one worker
                # picks up each checkpoint after preemption.
                if not try_claim(lock_dir, rid + ".resume", args.cluster):
                    print(f"[sweep] {rid} resume already claimed, trying next...")
                    continue
                print(
                    f"[sweep] Resuming {rid} (iter {c['iter_num']}) "
                    f"[{_override_summary(c['overrides'])}]"
                )
                run_training(c["config"], rid, c["out_dir"], args.cluster, args.wandb_buffer)
                # Remove resume lock so a future preemption cycle can re-claim
                resume_lock = os.path.join(lock_dir, f"{rid}.resume.lock")
                try:
                    os.remove(resume_lock)
                except FileNotFoundError:
                    pass
                claimed = True
                break

            if try_claim(lock_dir, rid, args.cluster):
                print(
                    f"[sweep] Claimed {rid} [{_override_summary(c['overrides'])}]"
                )
                run_training(c["config"], rid, c["out_dir"], args.cluster, args.wandb_buffer)
                claimed = True
                break
            else:
                print(f"[sweep] {rid} already claimed, trying next...")

        if not claimed:
            print("[sweep] Could not claim any config. Grid exhausted or all claimed.")
            break

        # Loop back to re-scan for next config
        print("[sweep] Training complete. Scanning for next config...")


def print_status(args):
    """Print table: cfg_hash | key params | cluster | wandb state | iter/max_iters | status"""
    base_toml, project, group, parameters = parse_sweep_yaml(args.sweep)
    grid = expand_grid(parameters)

    configs = []
    all_run_ids = []
    for overrides in grid:
        config = build_config(base_toml, overrides)
        run_id = deterministic_run_id(config)
        out_dir = deterministic_out_dir(config, project, group)
        all_run_ids.append(run_id)
        configs.append({
            "config": config,
            "run_id": run_id,
            "out_dir": out_dir,
            "overrides": overrides,
        })

    wandb_states = query_wandb_status(project, all_run_ids)

    # Build header from parameter keys
    param_keys = list(parameters.keys())
    short_keys = [k.split(".")[-1] for k in param_keys]

    # Column widths
    hash_w = 10
    param_ws = [max(len(sk), 8) for sk in short_keys]
    cluster_w = 8
    state_w = 10
    progress_w = 20

    # Print header
    header = f"{'cfg_hash':<{hash_w}}"
    for sk, pw in zip(short_keys, param_ws):
        header += f" | {sk:<{pw}}"
    header += f" | {'cluster':<{cluster_w}} | {'state':<{state_w}} | {'progress':<{progress_w}}"
    print(header)
    print("-" * len(header))

    for c in configs:
        rid = c["run_id"]
        ws = wandb_states.get(rid, {})
        state = ws.get("state", "-")
        cluster = ws.get("cluster", "-")

        has_ckpt, iter_num = check_local_checkpoint(c["out_dir"])
        max_iters = int(c["config"].training.max_iters)

        if state == "finished":
            progress = f"{max_iters}/{max_iters}"
        elif has_ckpt:
            progress = f"{iter_num}/{max_iters}"
        else:
            # Check lock file
            lock_dir = os.path.join(STORAGE_ROOT, "out", slug(project), ".locks")
            lock_path = os.path.join(lock_dir, f"{rid}.lock")
            if os.path.exists(lock_path):
                progress = "claimed"
            else:
                progress = "unclaimed"

        row = f"{rid:<{hash_w}}"
        for pk, pw in zip(param_keys, param_ws):
            val = c["overrides"].get(pk, "-")
            row += f" | {str(val):<{pw}}"
        row += f" | {cluster:<{cluster_w}} | {state:<{state_w}} | {progress:<{progress_w}}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Sweep runner for grid search coordination")
    parser.add_argument("--sweep", required=True, help="Path to sweep YAML file")
    parser.add_argument("--cluster", default="local", help="Cluster name (e.g., slurm, local)")
    parser.add_argument("--wandb-buffer", action="store_true",
                        help="Buffer wandb logs until checkpoint (for preemptible jobs)")
    parser.add_argument("--status", action="store_true",
                        help="Show status table and exit")
    parser.add_argument("--allow-behind", action="store_true",
                        help="Allow running even if local branch is behind remote")
    args = parser.parse_args()

    if not args.status and not args.allow_behind:
        ok, msg = check_git_is_current()
        if not ok:
            print(f"[sweep] ABORT: {msg}", file=sys.stderr)
            print("[sweep] Pull the latest changes or pass --allow-behind to override.", file=sys.stderr)
            sys.exit(1)
        print(f"[sweep] Git check: {msg}")

    if args.status:
        print_status(args)
    else:
        main_loop(args)


if __name__ == "__main__":
    main()
