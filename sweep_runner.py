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
import atexit
import itertools
import json
import os
import random
import signal
import socket
import subprocess
import sys
import threading
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


# ---------------------------------------------------------------------------
# Worker heartbeat
# ---------------------------------------------------------------------------

def _detect_gpu_type() -> str:
    """Return GPU model name (e.g. 'H100', 'B200') via nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            name = out.stdout.strip().splitlines()[0]
            # Extract short model: "NVIDIA H100 80GB HBM3" -> "H100"
            for token in name.split():
                if token[0].isalpha() and any(c.isdigit() for c in token):
                    return token
            return name.split()[-1] if name else "unknown"
    except Exception:
        pass
    return "unknown"


def _worker_id() -> str:
    """Return a unique worker identifier: hostname:gpuN."""
    host = socket.gethostname()
    cuda_dev = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    # SLURM_LOCALID is the per-node task index (0-7)
    local_id = os.environ.get("SLURM_LOCALID", cuda_dev)
    return f"{host}:gpu{local_id}"


class WorkerHeartbeat:
    """Background thread that writes periodic heartbeat JSON files."""

    INTERVAL = 30  # seconds between heartbeat writes

    def __init__(self, heartbeat_dir: str):
        self._dir = heartbeat_dir
        self._wid = _worker_id()
        self._path = os.path.join(heartbeat_dir, f"{self._wid}.json")
        self._gpu_type = _detect_gpu_type()
        self._slurm_job_id = _get_slurm_job_id()
        self._lock = threading.Lock()
        self._state = "starting"
        self._run_id: str | None = None
        self._config_label: str | None = None
        self._out_dir: str | None = None
        self._max_iters = 0
        self._last_error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        os.makedirs(heartbeat_dir, exist_ok=True)
        self._write()
        self._thread.start()
        atexit.register(self.shutdown)

    @property
    def worker_id(self) -> str:
        return self._wid

    def set_scanning(self):
        with self._lock:
            self._state = "scanning"
            self._run_id = None
            self._config_label = None
            self._out_dir = None
            self._max_iters = 0
        self._write()

    def set_training(self, run_id: str, config_label: str, out_dir: str, max_iters: int):
        with self._lock:
            self._state = "training"
            self._run_id = run_id
            self._config_label = config_label
            self._out_dir = out_dir
            self._max_iters = max_iters
        self._write()

    def set_error(self, msg: str):
        with self._lock:
            self._last_error = msg
            self._state = "error"
        self._write()

    def set_idle(self):
        with self._lock:
            self._state = "idle"
            self._run_id = None
            self._config_label = None
            self._out_dir = None
            self._max_iters = 0
        self._write()

    def _read_iter_num(self, out_dir: str | None) -> int:
        """Read current iter_num from the training run's latest checkpoint."""
        if not out_dir:
            return 0
        ts_path = os.path.join(out_dir, "checkpoints", "latest", "trainer_state.json")
        try:
            with open(ts_path) as f:
                return json.load(f).get("iter_num", 0)
        except Exception:
            return 0

    def _write(self):
        """Write heartbeat JSON atomically."""
        with self._lock:
            state = self._state
            out_dir = self._out_dir
            data = {
                "worker_id": self._wid,
                "slurm_job_id": self._slurm_job_id,
                "gpu_type": self._gpu_type,
                "state": state,
                "run_id": self._run_id,
                "config_label": self._config_label,
                "max_iters": self._max_iters,
                "last_error": self._last_error,
                "timestamp": time.time(),
            }
        # Read iter_num outside the lock (file I/O)
        data["iter_num"] = self._read_iter_num(out_dir) if state == "training" else 0
        tmp = self._path + f".tmp.{os.getpid()}"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self._path)
        except OSError:
            pass

    def _loop(self):
        while not self._stop.wait(self.INTERVAL):
            self._write()

    def shutdown(self):
        """Stop the heartbeat thread and remove the heartbeat file."""
        self._stop.set()
        self._thread.join(timeout=5)
        try:
            os.remove(self._path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Preemption monitor (GPU rescheduling)
# ---------------------------------------------------------------------------

class PreemptionMonitor:
    """Background thread that polls for preemption signal files from the autoscaler.

    The autoscaler writes .preempt/{worker_id_safe}.json when it wants to
    migrate a run to a better GPU.  When detected, this monitor sends SIGUSR1
    to our own process (triggering the handler in train.py) and deletes the
    signal file.
    """

    POLL_INTERVAL = 5  # seconds

    def __init__(self, project_slug: str, worker_id: str):
        self._project_slug = project_slug
        self._worker_id = worker_id
        safe = _worker_id_to_filename(worker_id)
        self._signal_path = os.path.join(
            STORAGE_ROOT, "out", project_slug, ".preempt", f"{safe}.json"
        )
        self.was_preempted = False
        self._reason: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.wait(self.POLL_INTERVAL):
            if not os.path.exists(self._signal_path):
                continue
            try:
                with open(self._signal_path) as f:
                    data = json.load(f)
                self._reason = data.get("reason", "gpu_upgrade")
                print(f"[preempt-monitor] Preemption signal detected: {self._reason}")
            except (json.JSONDecodeError, OSError) as e:
                print(f"[preempt-monitor] Error reading signal file: {e}")
                self._reason = "gpu_upgrade"
            # Delete the signal file before sending the signal
            try:
                os.remove(self._signal_path)
            except OSError:
                pass
            self.was_preempted = True
            os.kill(os.getpid(), signal.SIGUSR1)
            break  # stop polling after preemption

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    @property
    def reason(self) -> str | None:
        return self._reason


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
        lock_data = {
            "cluster": cluster,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid(),
        }
        if cluster == "slurm":
            slurm_id = _get_slurm_job_id()
            if slurm_id:
                lock_data["slurm_job_id"] = slurm_id
        content = json.dumps(lock_data)
        os.write(fd, content.encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _get_slurm_job_id() -> str | None:
    """Return the current Slurm job ID, or None if not in a Slurm environment.

    For array jobs returns "ARRAY_JOB_ID_TASK_ID" (e.g. "12345_3").
    """
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    if array_job:
        task = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
        return f"{array_job}_{task}"
    return os.environ.get("SLURM_JOB_ID")


def _read_lock_file(lock_dir: str, run_id: str) -> dict | None:
    """Read and parse a lock file. Returns None if missing or malformed."""
    try:
        with open(os.path.join(lock_dir, f"{run_id}.lock")) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _check_slurm_jobs_alive(job_ids: set[str]) -> set[str]:
    """Check which Slurm job IDs are still alive via squeue.

    Returns the subset of job_ids that are still running/pending.
    Fail-open: on any error, returns all job_ids (no false reclaims).
    """
    if not job_ids:
        return set()
    try:
        result = subprocess.run(
            ["squeue", "--noheader", "-o", "%i", "-j", ",".join(job_ids)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        alive = set()
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if line:
                alive.add(line)
        return alive
    except Exception:
        # Fail-open: assume all alive if squeue is unavailable
        return set(job_ids)


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
    try:
        train.main(config)
    except Exception as e:
        print(f"[sweep] Training failed for {run_id}: {e}")
        raise
    finally:
        # Clean up env vars regardless of success/failure
        os.environ.pop("RUN_ID", None)
        os.environ.pop("OUT_DIR", None)


def override_summary(overrides: dict) -> str:
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

    # Start worker heartbeat
    heartbeat_dir = os.path.join(STORAGE_ROOT, "out", slug(project), ".heartbeats")
    heartbeat = WorkerHeartbeat(heartbeat_dir)
    print(f"[sweep] Worker {heartbeat.worker_id} heartbeat -> {heartbeat_dir}")

    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 5

    while True:
        heartbeat.set_scanning()
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

        # Pre-scan: collect Slurm job IDs from lock files of "running" runs
        # AND from resume locks (to detect dead jobs blocking resumption).
        slurm_job_ids_to_check: set[str] = set()
        lock_data_cache: dict[str, dict] = {}
        resume_lock_cache: dict[str, dict] = {}
        for c in candidates:
            rid = c["run_id"]
            if wandb_states.get(rid, {}).get("state") == "running":
                lock_info = _read_lock_file(lock_dir, rid)
                if lock_info:
                    lock_data_cache[rid] = lock_info
                    sjid = lock_info.get("slurm_job_id")
                    if sjid:
                        slurm_job_ids_to_check.add(sjid)
            # Also check resume locks for stale Slurm jobs
            resume_info = _read_lock_file(lock_dir, rid + ".resume")
            if resume_info:
                resume_lock_cache[rid] = resume_info
                sjid = resume_info.get("slurm_job_id")
                if sjid:
                    slurm_job_ids_to_check.add(sjid)

        alive_slurm_jobs: set[str] = set()
        if slurm_job_ids_to_check and args.cluster == "slurm":
            alive_slurm_jobs = _check_slurm_jobs_alive(slurm_job_ids_to_check)

        # Remove stale resume locks whose Slurm jobs are dead
        if args.cluster == "slurm":
            for rid, rinfo in resume_lock_cache.items():
                sjid = rinfo.get("slurm_job_id")
                if sjid and sjid not in alive_slurm_jobs:
                    resume_path = os.path.join(lock_dir, f"{rid}.resume.lock")
                    print(f"[sweep] {rid}: resume lock held by dead slurm job {sjid} — removing")
                    try:
                        os.remove(resume_path)
                    except OSError:
                        pass
                elif not sjid:
                    # Legacy lock without slurm_job_id — use age-based cleanup
                    resume_path = os.path.join(lock_dir, f"{rid}.resume.lock")
                    try:
                        lock_age = time.time() - os.path.getmtime(resume_path)
                    except OSError:
                        lock_age = 0
                    if lock_age > 2 * 3600:  # 2 hours
                        print(f"[sweep] {rid}: resume lock without job ID, age {lock_age/3600:.1f}h — removing")
                        try:
                            os.remove(resume_path)
                        except OSError:
                            pass

        n_done = 0
        n_running = 0
        for c in candidates:
            rid = c["run_id"]
            ws = wandb_states.get(rid, {})
            state = ws.get("state")

            # Skip finished runs
            if state == "finished":
                n_done += 1
                continue

            has_ckpt, iter_num = check_local_checkpoint(c["out_dir"])
            c["has_ckpt"] = has_ckpt
            c["iter_num"] = iter_num

            if has_ckpt and state != "running":
                # Skip if checkpoint shows training already finished
                max_iters = int(c["config"].training.max_iters)
                if iter_num >= max_iters:
                    n_done += 1
                    continue
                # Also check if final checkpoint exists on disk (latest symlink may be stale)
                final_step_name = f"step-{max_iters:06d}"
                final_step = os.path.join(c["out_dir"], "checkpoints", final_step_name)
                if os.path.isdir(final_step):
                    print(f"[sweep] {rid}: latest symlink stale but {final_step_name} exists — skipping (done)")
                    n_done += 1
                    continue
                # Local partial checkpoint — resume it
                resumable.append(c)
            elif state == "running":
                # Check if the Slurm job that owns this run is dead
                lock_info = lock_data_cache.get(rid)
                slurm_dead = False
                if lock_info:
                    sjid = lock_info.get("slurm_job_id")
                    if sjid and sjid not in alive_slurm_jobs:
                        slurm_dead = True
                if slurm_dead:
                    print(f"[sweep] {rid}: wandb=running but slurm job {sjid} is dead — reclaiming")
                    # Remove stale lock so try_claim() can create a fresh one
                    try:
                        os.remove(os.path.join(lock_dir, f"{rid}.lock"))
                    except OSError:
                        pass
                    if has_ckpt:
                        resumable.append(c)
                    else:
                        claimable.append(c)
                else:
                    # Truly running or no slurm info — skip as before
                    n_running += 1
                    continue
            elif state in ("crashed", "failed"):
                # Failed/crashed — reclaimable
                claimable.append(c)
            elif state is None:
                # Not in wandb at all — new work
                lock_path = os.path.join(lock_dir, f"{rid}.lock")
                if os.path.exists(lock_path):
                    # Stale lock recovery: if lock exists but no checkpoint
                    # was ever saved and the lock is old, the claiming worker
                    # likely OOM'd or crashed before making progress.  Remove
                    # the lock so another worker can pick it up.
                    if not has_ckpt:
                        try:
                            lock_age = time.time() - os.path.getmtime(lock_path)
                        except OSError:
                            lock_age = 0
                        if lock_age > 30 * 60:  # 30 minutes
                            print(f"[sweep] Removing stale lock for {rid} "
                                  f"(age {lock_age/60:.0f}m, no checkpoint)")
                            try:
                                os.remove(lock_path)
                            except OSError:
                                pass
                        else:
                            continue
                    else:
                        continue
                claimable.append(c)

        # Priority: resume first, then new work
        ordered = resumable + claimable

        if not ordered:
            if n_done == len(candidates):
                print(f"[sweep] All {n_done} configs are done. Exiting.")
                heartbeat.set_idle()
                break
            heartbeat.set_idle()
            print(f"[sweep] No candidates available ({n_done} done, {n_running} running, "
                  f"{len(candidates) - n_done - n_running} locked). Sleeping 120s...")
            time.sleep(120)
            continue

        # Shuffle to reduce collision across parallel GPU workers
        random.shuffle(ordered)
        # But put resumable ones first (they were shuffled among themselves)
        ordered.sort(key=lambda c: not c.get("has_ckpt", False))

        attempted = False
        trained_ok = False
        for c in ordered:
            rid = c["run_id"]
            if c.get("has_ckpt"):
                # Lock file from the original claim already exists — that's
                # expected.  Use a separate resume lock so only one worker
                # picks up each checkpoint after preemption.
                if not try_claim(lock_dir, rid + ".resume", args.cluster):
                    print(f"[sweep] {rid} resume already claimed, trying next...")
                    continue
                config_label = override_summary(c["overrides"])
                print(
                    f"[sweep] Resuming {rid} (iter {c['iter_num']}) "
                    f"[{config_label}]"
                )
                max_iters = int(c["config"].training.max_iters)
                heartbeat.set_training(rid, config_label, c["out_dir"], max_iters)
                resume_lock = os.path.join(lock_dir, f"{rid}.resume.lock")
                try:
                    run_training(c["config"], rid, c["out_dir"], args.cluster, args.wandb_buffer)
                    consecutive_failures = 0
                    trained_ok = True
                except Exception as e:
                    print(f"[sweep] Run {rid} failed: {e}")
                    heartbeat.set_error(str(e))
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f"[sweep] {MAX_CONSECUTIVE_FAILURES} consecutive failures — "
                              f"likely a systemic issue (broken proxy?). Exiting.")
                        sys.exit(1)
                    backoff = min(60 * (2 ** (consecutive_failures - 1)), 600)
                    print(f"[sweep] Backing off {backoff}s after {consecutive_failures} "
                          f"consecutive failure(s)")
                    time.sleep(backoff)
                finally:
                    # Always remove resume lock so a future cycle can re-claim
                    try:
                        os.remove(resume_lock)
                    except FileNotFoundError:
                        pass
                attempted = True
                break

            if try_claim(lock_dir, rid, args.cluster):
                config_label = override_summary(c["overrides"])
                print(
                    f"[sweep] Claimed {rid} [{config_label}]"
                )
                max_iters = int(c["config"].training.max_iters)
                heartbeat.set_training(rid, config_label, c["out_dir"], max_iters)
                try:
                    run_training(c["config"], rid, c["out_dir"], args.cluster, args.wandb_buffer)
                    consecutive_failures = 0
                    trained_ok = True
                except Exception as e:
                    print(f"[sweep] Run {rid} failed: {e}")
                    heartbeat.set_error(str(e))
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f"[sweep] {MAX_CONSECUTIVE_FAILURES} consecutive failures — "
                              f"likely a systemic issue (broken proxy?). Exiting.")
                        sys.exit(1)
                    backoff = min(60 * (2 ** (consecutive_failures - 1)), 600)
                    print(f"[sweep] Backing off {backoff}s after {consecutive_failures} "
                          f"consecutive failure(s)")
                    time.sleep(backoff)
                attempted = True
                break
            else:
                print(f"[sweep] {rid} already claimed, trying next...")

        if not attempted:
            heartbeat.set_idle()
            print("[sweep] Could not claim any config — all locked. Sleeping 120s before re-scan...")
            time.sleep(120)
            continue

        # Loop back to re-scan for next config
        if trained_ok:
            print("[sweep] Training complete. Scanning for next config...")
        else:
            print("[sweep] Training failed. Scanning for next config...")


# ---------------------------------------------------------------------------
# Poll-based worker mode (centralized assignment from autoscaler)
# ---------------------------------------------------------------------------

def _worker_id_to_filename(worker_id: str) -> str:
    """Convert worker_id to a safe filename (replace ':' with '_')."""
    return worker_id.replace(":", "_")


def _read_assignment(path: str) -> dict | None:
    """Read an assignment JSON file. Returns None if missing or malformed."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _delete_assignment(path: str):
    """Delete an assignment file (signals completion to autoscaler)."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def poll_loop(args):
    """Poll-based main loop: wait for autoscaler to assign work via .assignments/ files."""
    base_toml, project, group, parameters = parse_sweep_yaml(args.sweep)
    project_slug = slug(project)

    # Start heartbeat
    heartbeat_dir = os.path.join(STORAGE_ROOT, "out", project_slug, ".heartbeats")
    heartbeat = WorkerHeartbeat(heartbeat_dir)
    wid = heartbeat.worker_id

    assignment_dir = os.path.join(STORAGE_ROOT, "out", project_slug, ".assignments")
    os.makedirs(assignment_dir, exist_ok=True)
    assignment_path = os.path.join(assignment_dir, f"{_worker_id_to_filename(wid)}.json")

    print(f"[sweep-poll] Worker {wid} polling {assignment_path}")

    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 5
    POLL_INTERVAL = 10  # seconds

    heartbeat.set_idle()

    while True:
        # Check for assignment file
        assignment = _read_assignment(assignment_path)

        if assignment is None:
            time.sleep(POLL_INTERVAL)
            continue

        # Check for shutdown signal
        if assignment.get("action") == "shutdown":
            print(f"[sweep-poll] Received shutdown signal. Exiting.")
            _delete_assignment(assignment_path)
            heartbeat.set_idle()
            break

        # We have a training assignment
        run_id = assignment["run_id"]
        config_label = assignment.get("config_label", run_id)
        out_dir = assignment["out_dir"]
        overrides = assignment["overrides"]

        config = build_config(assignment.get("base_toml", base_toml), overrides)
        max_iters = int(config.training.max_iters)

        print(f"[sweep-poll] Assigned {run_id} [{config_label}]")
        heartbeat.set_training(run_id, config_label, out_dir, max_iters)

        # Start preemption monitor (watches for .preempt/ signal files)
        monitor = PreemptionMonitor(project_slug, wid)

        try:
            run_training(config, run_id, out_dir, args.cluster, args.wandb_buffer)
            consecutive_failures = 0
            if monitor.was_preempted:
                print(f"[sweep-poll] Preempted {run_id} for GPU upgrade — deleting assignment, going idle")
            else:
                print(f"[sweep-poll] Training complete for {run_id}")
        except Exception as e:
            if monitor.was_preempted:
                print(f"[sweep-poll] Preempted {run_id} for GPU upgrade (exited with error: {e})")
            else:
                print(f"[sweep-poll] Training failed for {run_id}: {e}")
                heartbeat.set_error(str(e))
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"[sweep-poll] {MAX_CONSECUTIVE_FAILURES} consecutive failures. Exiting.")
                    _delete_assignment(assignment_path)
                    monitor.stop()
                    sys.exit(1)
                backoff = min(60 * (2 ** (consecutive_failures - 1)), 600)
                print(f"[sweep-poll] Backing off {backoff}s after {consecutive_failures} "
                      f"consecutive failure(s)")
                time.sleep(backoff)
        finally:
            monitor.stop()

        # Delete assignment and go idle
        _delete_assignment(assignment_path)
        heartbeat.set_idle()
        print(f"[sweep-poll] Idle, waiting for next assignment...")


def print_status(args):
    """Print table: cfg_hash | key params | cluster | slurm_job | wandb state | iter/max_iters | status"""
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

    # Read lock files and collect Slurm job IDs
    lock_dir = os.path.join(STORAGE_ROOT, "out", slug(project), ".locks")
    lock_data_map: dict[str, dict] = {}
    slurm_job_ids_to_check: set[str] = set()
    for c in configs:
        lock_info = _read_lock_file(lock_dir, c["run_id"])
        if lock_info:
            lock_data_map[c["run_id"]] = lock_info
            sjid = lock_info.get("slurm_job_id")
            if sjid:
                slurm_job_ids_to_check.add(sjid)

    # Batch check which Slurm jobs are alive (fail-open if squeue unavailable)
    alive_slurm_jobs: set[str] = set()
    if slurm_job_ids_to_check:
        alive_slurm_jobs = _check_slurm_jobs_alive(slurm_job_ids_to_check)

    # Build header from parameter keys
    param_keys = list(parameters.keys())
    short_keys = [k.split(".")[-1] for k in param_keys]

    # Column widths
    hash_w = 10
    param_ws = [max(len(sk), 8) for sk in short_keys]
    cluster_w = 8
    slurm_job_w = 12
    state_w = 12
    progress_w = 20

    # Print header
    header = f"{'cfg_hash':<{hash_w}}"
    for sk, pw in zip(short_keys, param_ws):
        header += f" | {sk:<{pw}}"
    header += (f" | {'cluster':<{cluster_w}} | {'slurm_job':<{slurm_job_w}}"
               f" | {'state':<{state_w}} | {'progress':<{progress_w}}")
    print(header)
    print("-" * len(header))

    for c in configs:
        rid = c["run_id"]
        ws = wandb_states.get(rid, {})
        state = ws.get("state", "-")
        cluster = ws.get("cluster", "-")

        lock_info = lock_data_map.get(rid, {})
        slurm_job = lock_info.get("slurm_job_id", "-")

        # Override state to dead-slurm if wandb says running but Slurm job is dead
        if state == "running" and slurm_job != "-" and slurm_job not in alive_slurm_jobs:
            state = "dead-slurm"

        has_ckpt, iter_num = check_local_checkpoint(c["out_dir"])
        max_iters = int(c["config"].training.max_iters)

        if state == "finished":
            progress = f"{max_iters}/{max_iters}"
        elif has_ckpt:
            progress = f"{iter_num}/{max_iters}"
        else:
            lock_path = os.path.join(lock_dir, f"{rid}.lock")
            if os.path.exists(lock_path):
                progress = "claimed"
            else:
                progress = "unclaimed"

        row = f"{rid:<{hash_w}}"
        for pk, pw in zip(param_keys, param_ws):
            val = c["overrides"].get(pk, "-")
            row += f" | {str(val):<{pw}}"
        row += (f" | {cluster:<{cluster_w}} | {str(slurm_job):<{slurm_job_w}}"
                f" | {state:<{state_w}} | {progress:<{progress_w}}")
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
    parser.add_argument("--mode", choices=["lock", "poll"], default="lock",
                        help="Coordination mode: 'lock' (legacy file locks) or 'poll' (autoscaler assigns work)")
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
    elif args.mode == "poll":
        poll_loop(args)
    else:
        main_loop(args)


if __name__ == "__main__":
    main()
