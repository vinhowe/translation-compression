#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "rich",
#     "pyyaml",
# ]
# ///
"""
sweep-dashboard.py — Rich TUI for monitoring sweep_runner progress.

Usage:
    ./scripts/sweep-dashboard.py sweeps/bpe16384-n3-n5.yaml
    ./scripts/sweep-dashboard.py sweeps/bpe16384-n3-n5.yaml --refresh 10
    ./scripts/sweep-dashboard.py sweeps/bpe16384-n3-n5.yaml --once   # single snapshot
"""

import argparse
import hashlib
import itertools
import json
import os
import re
import subprocess
import time
from pathlib import Path

import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
TC_STORAGE_ROOT = os.environ.get(
    "TC_STORAGE_ROOT",
    "/nobackup/archive/grp/grp_pccl/vin/dev/translation-compression",
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGDIR = PROJECT_ROOT / "logs"
SIZE_ORDER = {"8-32": 0, "8-64": 1, "8-128": 2, "8-256": 3}
MAX_GPUS_PER_NODE = int(os.environ.get("TC_SWEEP_DASH_MAX_GPUS", "8"))


# ---------------------------------------------------------------------------
# Sweep parsing
# ---------------------------------------------------------------------------

def parse_sweep(path: str):
    with open(path) as f:
        spec = yaml.safe_load(f)
    return spec.get("project", "translation-compression"), spec.get("name", "sweep"), spec.get("parameters", {})


def expand_grid(params: dict) -> list[dict]:
    keys = sorted(params.keys())
    vals = [params[k]["values"] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _cfg_hash_from_dict(d: dict) -> str:
    """Compute cfg_hash from a config dict (same algo as src.experiment.cfg_hash).

    Normalises wandb_buffer to False so the hash matches what sweep_runner
    computed *before* run_training() toggled the flag.
    """
    d = json.loads(json.dumps(d))  # deep copy
    if "logging" in d and isinstance(d["logging"], dict):
        d["logging"]["wandb_buffer"] = False
    s = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2s(s.encode(), digest_size=4).hexdigest()


def scan_runs(project: str, group: str):
    """Scan output dirs. Return {rid: {iter_num, max_iters, mtime, size_tier, ...}}.

    The rid is the canonical cfg_hash recomputed from config.json, so old-format
    dirs (pre-deterministic run IDs) map to the same grid slot as new ones.
    When cfg_hash can't be computed, falls back to the directory name prefix.
    """
    base = Path(TC_STORAGE_ROOT) / "out" / _slug(project) / _slug(group)
    if not base.exists():
        return {}

    runs: dict[str, dict] = {}
    for d in base.iterdir():
        if not d.is_dir() or "_s" not in d.name:
            continue
        dir_rid = d.name.split("_s")[0]

        # Checkpoint info
        iter_num, mtime = 0, 0.0
        state_file = d / "checkpoints" / "latest" / "trainer_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    st = json.load(f)
                iter_num = st.get("iter_num", 0)
                mtime = state_file.stat().st_mtime
            except (json.JSONDecodeError, OSError):
                pass

        # Meta config
        max_iters = 1_000_000
        size_tier = ""
        n_comp = 0
        ratio_mode = ""
        ratio = 0.0
        rid = dir_rid  # default: use directory name

        cfg_file = d / "meta" / "config.json"
        if cfg_file.exists():
            try:
                with open(cfg_file) as f:
                    cfg = json.load(f)
                max_iters = cfg.get("training", {}).get("max_iters", max_iters)
                size_tier = str(cfg.get("model", {}).get("size_tier", ""))
                n_comp = cfg.get("experiment", {}).get("n_compartments", 0)
                ratio_mode = cfg.get("experiment", {}).get("translation_ratio_mode", "")
                ratio = cfg.get("experiment", {}).get("translation_ratio", 0.0)
                rid = _cfg_hash_from_dict(cfg)
            except (json.JSONDecodeError, OSError):
                pass

        # If latest symlink is stale, check for the final step checkpoint
        if iter_num < max_iters:
            final = d / "checkpoints" / f"step-{int(max_iters)}"
            if final.is_dir():
                final_state = final / "trainer_state.json"
                if final_state.exists():
                    try:
                        with open(final_state) as f:
                            st = json.load(f)
                        iter_num = st.get("iter_num", iter_num)
                        mtime = final_state.stat().st_mtime
                    except (json.JSONDecodeError, OSError):
                        pass

        # Lock status — check both canonical rid and original dir rid
        locks_dir = base.parent / ".locks"
        locked = (locks_dir / f"{rid}.lock").exists() or (locks_dir / f"{dir_rid}.lock").exists()

        # If two dirs map to the same cfg_hash, keep the one with more progress
        if rid in runs and runs[rid]["iter_num"] >= iter_num:
            continue

        runs[rid] = {
            "rid": rid,
            "dir_rid": dir_rid,
            "iter_num": iter_num,
            "max_iters": max_iters,
            "mtime": mtime,
            "size_tier": size_tier,
            "n_compartments": n_comp,
            "ratio_mode": ratio_mode,
            "translation_ratio": ratio,
            "locked": locked,
        }
    return runs


def _parse_override_summary(text: str) -> dict[str, str]:
    """Parse 'key=val key=val' summary into dict."""
    out: dict[str, str] = {}
    for part in text.split():
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k] = v
    return out


def _to_int(val) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _to_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def get_running_map(
    done_rids: set[str] | None = None,
    lock_dir: Path | None = None,
) -> tuple[dict[str, dict], int, int, dict[str, dict[str, str]]]:
    """Return ({rid: {"jobid", "node", "gpu"}}, num_running, num_pending, rid_to_overrides).

    *done_rids*, if provided, are skipped during assignment so that workers
    which finished a config get mapped to whatever they picked up next.

    *lock_dir*, if provided, enables backfill using resume locks: after the
    greedy assignment, any job with unfilled GPU slots gets assigned rids
    whose ``.resume.lock`` has been held for >5 min (actively training).
    """
    if done_rids is None:
        done_rids = set()
    result = subprocess.run(
        ["squeue", "--me", "--noheader", "-o", "%i|%t|%N"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {}, 0, 0, {}

    n_run = n_pend = 0
    running_jobids = []
    job_to_node: dict[str, str] = {}
    for line in result.stdout.strip().splitlines():
        parts = line.strip().split("|")
        if len(parts) < 2:
            continue
        jid, state = parts[0], parts[1]
        node = parts[2] if len(parts) > 2 else ""
        if state == "R":
            n_run += 1
            running_jobids.append(jid)
            job_to_node[jid] = node
        elif state == "PD":
            n_pend += 1

    # Get GPU types for running nodes via scontrol show node
    node_gpu: dict[str, str] = {}
    nodes = {n for n in job_to_node.values() if n}
    if nodes:
        try:
            raw = subprocess.run(
                ["scontrol", "show", "node", ",".join(sorted(nodes))],
                capture_output=True, text=True, timeout=10,
            ).stdout
            current_node = ""
            for line in raw.splitlines():
                m = re.search(r"NodeName=(\S+)", line)
                if m:
                    current_node = m.group(1)
                if re.match(r"Gres=", line.strip()) or "   Gres=" in line:
                    gm = re.search(r"gpu:(\w+):\d+", line)
                    if gm and "Used" not in line:
                        node_gpu[current_node] = gm.group(1).upper()
        except Exception:
            pass

    rid_to_job: dict[str, dict] = {}
    rid_to_overrides: dict[str, dict[str, str]] = {}

    # For each job, collect the unique rids it has claimed/resumed (newest
    # first) and how many training completions it has logged.  The number
    # of active workers is  claims − completions  (capped at GPU count).
    #
    # To assign each rid to exactly one job we use a greedy algorithm:
    # process jobs in order of fewest completions first (most stable).
    # Stable jobs rarely cycle, so their "last N unique" rids are accurate.
    # Heavily-cycling jobs go last and pick up whatever remains.
    _CLAIM_RE = re.compile(
        r"\[sweep\] (?:Claimed|Resuming) (?P<rid>[a-f0-9]{8})"
        r"(?:[^\[]*\[(?P<summary>[^\]]+)\])?"
    )
    _COMPLETE_RE = re.compile(r"\[sweep\] Training complete\.")

    job_candidates: dict[str, list[str]] = {}   # jobid -> rids newest-first
    job_active_count: dict[str, int] = {}
    job_completions: dict[str, int] = {}

    for jobid in running_jobids:
        for logfile in sorted(LOGDIR.glob(f"*-{jobid}*.out")):
            try:
                text = logfile.read_text(errors="replace")
            except OSError:
                continue

            n_claims = 0
            claims: list[tuple[str, str | None]] = []
            for m in _CLAIM_RE.finditer(text):
                n_claims += 1
                claims.append((m.group("rid"), m.group("summary")))
            n_complete = len(_COMPLETE_RE.findall(text))

            if not claims:
                continue

            # Unique rids, newest first
            seen: set[str] = set()
            ordered: list[str] = []
            for rid, summary in reversed(claims):
                if rid in seen:
                    continue
                seen.add(rid)
                ordered.append(rid)
                if summary and rid not in rid_to_overrides:
                    rid_to_overrides[rid] = _parse_override_summary(summary)

            job_candidates[jobid] = ordered
            job_active_count[jobid] = min(max(0, n_claims - n_complete), MAX_GPUS_PER_NODE)
            job_completions[jobid] = n_complete

    # Greedy assignment: stable jobs (fewest completions) claim first
    assigned: set[str] = set()
    job_filled: dict[str, int] = {}
    for jobid in sorted(running_jobids, key=lambda j: job_completions.get(j, 0)):
        node = job_to_node.get(jobid, "")
        gpu = node_gpu.get(node, "")
        count = 0
        target = job_active_count.get(jobid, 0)
        for rid in job_candidates.get(jobid, []):
            if rid in assigned or rid in done_rids:
                continue
            rid_to_job[rid] = {"jobid": jobid, "node": node, "gpu": gpu}
            assigned.add(rid)
            count += 1
            if count >= target:
                break
        job_filled[jobid] = count

    # Backfill: if a job has unfilled GPU slots, use .resume.lock files to
    # identify configs that are actively training but didn't appear in that
    # job's log.  A resume lock held for >5 min means a worker is training
    # (short-lived locks are cycling workers briefly touching done configs).
    if lock_dir and lock_dir.is_dir():
        now = time.time()
        MIN_LOCK_AGE = 300  # 5 min — filters out cycling workers on done configs
        unfilled = {
            jid: job_active_count.get(jid, 0) - job_filled.get(jid, 0)
            for jid in running_jobids
            if job_filled.get(jid, 0) < job_active_count.get(jid, 0)
        }
        if unfilled:
            # Find rids with long-held resume locks that are unassigned/non-done
            candidates = []
            for lf in lock_dir.iterdir():
                if not lf.name.endswith(".resume.lock"):
                    continue
                rid = lf.name.removesuffix(".resume.lock")
                if rid in assigned or rid in done_rids:
                    continue
                try:
                    age = now - lf.stat().st_mtime
                except OSError:
                    continue
                if age >= MIN_LOCK_AGE:
                    candidates.append((rid, age))
            # Sort: longest-held first (most stable training)
            candidates.sort(key=lambda x: -x[1])

            for jobid in sorted(unfilled, key=lambda j: job_completions.get(j, 0)):
                node = job_to_node.get(jobid, "")
                gpu = node_gpu.get(node, "")
                remaining = unfilled[jobid]
                for rid, _ in candidates:
                    if rid in assigned:
                        continue
                    rid_to_job[rid] = {"jobid": jobid, "node": node, "gpu": gpu}
                    assigned.add(rid)
                    remaining -= 1
                    if remaining <= 0:
                        break

    return rid_to_job, n_run, n_pend, rid_to_overrides


# ---------------------------------------------------------------------------
# ETA
# ---------------------------------------------------------------------------

class ETATracker:
    def __init__(self):
        self._history: dict[str, list[tuple[float, int]]] = {}

    def update(self, rid: str, iter_num: int):
        if iter_num <= 0:
            return
        h = self._history.setdefault(rid, [])
        if not h or h[-1][1] != iter_num:
            h.append((time.time(), iter_num))
        if len(h) > 30:
            self._history[rid] = h[-30:]

    def eta_seconds(self, rid: str, max_iters: int) -> float | None:
        h = self._history.get(rid, [])
        if len(h) < 2:
            return None
        dt = h[-1][0] - h[0][0]
        di = h[-1][1] - h[0][1]
        if di <= 0 or dt <= 0:
            return None
        remaining = max_iters - h[-1][1]
        return max(0, remaining / (di / dt))

    def rate(self, rid: str) -> float | None:
        h = self._history.get(rid, [])
        if len(h) < 2:
            return None
        dt = h[-1][0] - h[0][0]
        di = h[-1][1] - h[0][1]
        if di <= 0 or dt <= 0:
            return None
        return di / dt


def fmt_eta(s: float | None) -> str:
    if s is None:
        return "—"
    if s <= 0:
        return "done"
    h = s / 3600
    if h < 1:
        return f"{s / 60:.0f}m"
    if h < 24:
        return f"{h:.1f}h"
    return f"{h / 24:.1f}d"


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _fmt_age(mtime: float) -> tuple[str, str]:
    """Return (text, style) for time since *mtime*, color-coded by recency."""
    if mtime <= 0:
        return ("", "bright_black")
    age = time.time() - mtime
    if age < 0:
        age = 0
    # Format
    if age < 60:
        txt = f"{int(age)}s"
    elif age < 3600:
        txt = f"{int(age / 60)}m"
    elif age < 86400:
        txt = f"{age / 3600:.1f}h"
    else:
        txt = f"{age / 86400:.1f}d"
    # Color
    if age < 60:
        sty = "bold green"
    elif age < 300:
        sty = "green"
    elif age < 1800:
        sty = "yellow"
    else:
        sty = "bold red"
    return (txt, sty)


def bar(pct: float, w: int = 12) -> Text:
    filled = max(0, min(w, int(pct / 100 * w)))
    t = Text()
    t.append("█" * filled, style="green")
    t.append("░" * (w - filled), style="bright_black")
    return t


GPU_STYLES = {"H100": "dark_orange", "H200": "medium_purple1", "B200": "bright_green"}


def read_heartbeats(project: str) -> list[dict]:
    """Read all worker heartbeat JSON files."""
    hb_dir = Path(TC_STORAGE_ROOT) / "out" / _slug(project) / ".heartbeats"
    if not hb_dir.is_dir():
        return []
    heartbeats = []
    for f in hb_dir.iterdir():
        if not f.name.endswith(".json") or ".tmp." in f.name:
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            data["_file_mtime"] = f.stat().st_mtime
            heartbeats.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return heartbeats


def build_workers_panel(heartbeats: list[dict]) -> Table | None:
    """Build a Rich Table showing worker status from heartbeat files."""
    if not heartbeats:
        return None

    now = time.time()

    # Classify workers
    n_training = n_scanning = n_idle = n_error = n_dead = 0
    for hb in heartbeats:
        age = now - hb.get("timestamp", 0)
        if age > 300:  # >5 min since last heartbeat = dead
            n_dead += 1
        elif hb["state"] == "training":
            n_training += 1
        elif hb["state"] == "scanning":
            n_scanning += 1
        elif hb["state"] == "error":
            n_error += 1
        else:
            n_idle += 1

    # Sort by node, then GPU index
    def sort_key(hb):
        wid = hb.get("worker_id", "")
        parts = wid.rsplit(":gpu", 1)
        host = parts[0] if parts else wid
        try:
            gpu_idx = int(parts[1]) if len(parts) > 1 else 0
        except ValueError:
            gpu_idx = 0
        return (host, gpu_idx)

    heartbeats_sorted = sorted(heartbeats, key=sort_key)

    tbl = Table(
        show_header=True, show_edge=False, show_lines=False,
        padding=(0, 1), expand=True,
        title=Text.assemble(
            ("Workers", "bold"),
            f"  {n_training} training  {n_scanning} scanning  "
            f"{n_idle} idle  {n_error} error  {n_dead} dead",
        ),
        title_style="",
    )
    tbl.add_column("Worker", style="cyan", no_wrap=True)
    tbl.add_column("GPU", no_wrap=True)
    tbl.add_column("State", no_wrap=True)
    tbl.add_column("Run", no_wrap=True)
    tbl.add_column("Progress", no_wrap=True)
    tbl.add_column("HB Age", no_wrap=True)
    tbl.add_column("Error", no_wrap=True)

    for hb in heartbeats_sorted:
        wid = hb.get("worker_id", "?")
        gpu_type = hb.get("gpu_type", "?")
        state = hb.get("state", "?")
        run_id = hb.get("run_id") or ""
        config_label = hb.get("config_label") or ""
        iter_num = hb.get("iter_num", 0)
        max_iters = hb.get("max_iters", 0)
        last_error = hb.get("last_error") or ""
        ts = hb.get("timestamp", 0)
        age = now - ts if ts > 0 else 999999

        # Dead detection
        is_dead = age > 300

        # GPU style
        gpu_sty = GPU_STYLES.get(gpu_type, "dim cyan")
        gpu_text = Text(gpu_type, style=gpu_sty)

        # State style
        if is_dead:
            state_text = Text("DEAD", style="bold red")
        elif state == "training":
            state_text = Text("training", style="bold green")
        elif state == "scanning":
            state_text = Text("scanning", style="yellow")
        elif state == "error":
            state_text = Text("ERROR", style="bold red")
        elif state == "idle":
            state_text = Text("idle", style="bright_black")
        else:
            state_text = Text(state, style="bright_black")

        # Run info
        run_text = Text()
        if run_id:
            run_text.append(run_id[:8], style="dim")
            if config_label:
                run_text.append(f" {config_label}", style="")

        # Progress
        progress_text = Text()
        if state == "training" and max_iters > 0 and not is_dead:
            pct = iter_num / max_iters * 100
            progress_text.append(f"{iter_num:,}/{max_iters:,}", style="")
            progress_text.append(f" {pct:.0f}%", style="bold")

        # Heartbeat age
        age_txt, age_sty = _fmt_age(ts)
        if is_dead:
            age_sty = "bold red"

        # Error (truncated)
        err_text = Text()
        if last_error:
            truncated = last_error[:40] + ("..." if len(last_error) > 40 else "")
            err_text.append(truncated, style="red")

        tbl.add_row(
            Text(wid, style="dim" if is_dead else "cyan"),
            gpu_text,
            state_text,
            run_text,
            progress_text,
            Text(age_txt, style=age_sty),
            err_text,
        )

    return tbl


def build(sweep_name, configs, rid_to_job, eta_tracker, n_jobs_run, n_jobs_pend,
          hide_done=False, heartbeats=None):
    total_iters = sum(c["max_iters"] for c in configs)
    done_iters = sum(c["iter_num"] for c in configs)
    pct = done_iters / total_iters * 100 if total_iters else 0

    n_done = sum(1 for c in configs if c["iter_num"] >= c["max_iters"])
    n_active = sum(1 for c in configs if c["rid"] in rid_to_job)
    # GPU counts from heartbeats (more accurate than scontrol)
    gpu_counts: dict[str, int] = {}
    if heartbeats:
        now = time.time()
        for hb in heartbeats:
            if now - hb.get("timestamp", 0) < 300 and hb.get("state") == "training":
                gt = hb.get("gpu_type", "unknown")
                gpu_counts[gt] = gpu_counts.get(gt, 0) + 1
    # Fallback to scontrol-based count if no heartbeats
    if not gpu_counts:
        n_h100 = sum(
            1 for c in configs
            if c["rid"] in rid_to_job and rid_to_job[c["rid"]].get("gpu", "") == "H100"
        )
        if n_h100:
            gpu_counts["H100"] = n_h100
    n_ckpt = sum(1 for c in configs if 0 < c["iter_num"] < c["max_iters"] and c["rid"] not in rid_to_job)
    n_unclaimed = sum(1 for c in configs if c["iter_num"] == 0 and not c["locked"])

    # Overall ETA
    etas = [eta_tracker.eta_seconds(c["rid"], c["max_iters"])
            for c in configs if c["rid"] in rid_to_job and c["iter_num"] > 0]
    etas = [e for e in etas if e is not None]
    if etas and n_active > 0:
        avg_eta = sum(etas) / len(etas)
        overall_eta = max(etas) + (n_unclaimed + n_ckpt) * avg_eta / max(n_active, 1)
    else:
        overall_eta = None

    hdr = Text()
    hdr.append(f" {sweep_name}", style="bold cyan")
    hdr.append(f"  {done_iters:,}/{total_iters:,}")
    hdr.append(f"  {pct:.1f}%")
    hdr.append(f"  ETA {fmt_eta(overall_eta)}", style="bold")
    hdr.append(f"  |  slurm: {n_jobs_run}R {n_jobs_pend}PD")
    gpu_parts = " ".join(f"{k}:{v}" for k, v in sorted(gpu_counts.items()))
    if gpu_parts:
        hdr.append(f"  |  {gpu_parts}")
    hdr.append(f"  |  {n_done}done {n_active}run {n_ckpt}wait {n_unclaimed}new")

    overall = Text(" ")
    overall.append_text(bar(pct, 60))
    overall.append(f" {pct:.1f}%", style="bold")

    # Sort configs (filter done from grid if requested)
    configs_s = sorted(configs, key=lambda c: (
        SIZE_ORDER.get(c["size_tier"], 99),
        c["n_compartments"],
        c["ratio_mode"],
        c["translation_ratio"],
    ))
    if hide_done:
        configs_s = [c for c in configs_s
                     if not (c["iter_num"] >= c["max_iters"] and c["max_iters"] > 0)]

    # Grid: 4 columns
    COLS = 4
    n = len(configs_s)
    rows = (n + COLS - 1) // COLS

    grid = Table(show_header=False, show_edge=False, show_lines=False,
                 padding=(0, 1), expand=True)
    for _ in range(COLS):
        grid.add_column(ratio=1)

    for r in range(rows):
        cells = []
        for col in range(COLS):
            idx = col * rows + r
            if idx >= n:
                cells.append(Text(""))
                continue

            c = configs_s[idx]
            rid = c["rid"]
            cpct = c["iter_num"] / c["max_iters"] * 100 if c["max_iters"] else 0
            running = rid in rid_to_job
            eta = eta_tracker.eta_seconds(rid, c["max_iters"])

            cell = Text()

            # Status marker
            if running:
                cell.append("R", style="bold green")
            elif c["iter_num"] >= c["max_iters"] and c["max_iters"] > 0:
                cell.append("D", style="bold cyan")
            elif c["iter_num"] > 0:
                cell.append("P", style="yellow")
            elif c["locked"]:
                cell.append("L", style="bright_black")
            else:
                cell.append(".", style="bright_black")

            # Label: size/nX/ratio  (X = A or C)
            m = c["ratio_mode"][:1].upper() if c["ratio_mode"] else "?"
            lbl = f"{c['size_tier']:>5}/{c['n_compartments']}{m}/{c['translation_ratio']}"
            style = "green" if running else ("yellow" if c["iter_num"] > 0 else "bright_black")
            cell.append(f" {lbl:<14}", style=style)

            # Bar
            cell.append_text(bar(cpct, 10))

            # Pct + node/GPU
            cell.append(f" {cpct:4.0f}%")
            if running:
                info = rid_to_job[rid]
                node = info.get("node", "")
                gpu = info.get("gpu", "")
                if node:
                    cell.append(f" {node}", style="dim")
                if gpu:
                    gpu_style = {"H100": "dark_orange", "H200": "medium_purple1", "B200": "bright_green"}.get(gpu, "dim cyan")
                    cell.append(f" {gpu}", style=gpu_style)

            # ETA
            if running and eta is not None:
                cell.append(f" {fmt_eta(eta)}", style="cyan")
            elif cpct >= 100:
                cell.append(" done", style="bold green")

            # Last checkpoint update
            if c["mtime"] > 0:
                age_txt, age_sty = _fmt_age(c["mtime"])
                if not running:
                    age_sty = "bright_black"
                cell.append(f" ↑{age_txt}", style=age_sty)

            cells.append(cell)
        grid.add_row(*cells)

    legend = Text(" ")
    legend.append("R", style="bold green")
    legend.append("=run ")
    legend.append("P", style="yellow")
    legend.append("=pause ")
    legend.append("D", style="bold cyan")
    legend.append("=done ")
    legend.append("L", style="bright_black")
    legend.append("=lock ")
    legend.append(".", style="bright_black")
    legend.append("=new  ")
    legend.append(time.strftime("%H:%M:%S"), style="bright_black")

    parts = [hdr, overall, Text(""), grid, Text(""), legend]

    # Workers panel (from heartbeats)
    if heartbeats:
        workers_tbl = build_workers_panel(heartbeats)
        if workers_tbl:
            parts.extend([Text(""), workers_tbl])

    return Group(*parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect(project, group, grid_overrides, eta_tracker):
    runs = scan_runs(project, group)
    done_rids = {
        rid for rid, info in runs.items()
        if info["max_iters"] > 0 and info["iter_num"] >= info["max_iters"]
    }
    lock_dir = Path(TC_STORAGE_ROOT) / "out" / _slug(project) / ".locks"
    rid_to_job, n_run, n_pend, rid_to_overrides = get_running_map(done_rids, lock_dir)

    # Build config list from filesystem runs
    configs = []
    seen = set()
    for rid, info in runs.items():
        eta_tracker.update(rid, info["iter_num"])
        configs.append(info)
        seen.add(rid)

    # Include running runs that haven't created output dirs yet
    base = Path(TC_STORAGE_ROOT) / "out" / _slug(project)
    lock_dir = base / ".locks"
    for rid, info in rid_to_job.items():
        if rid in seen:
            continue
        ov = rid_to_overrides.get(rid, {})
        configs.append({
            "rid": rid,
            "iter_num": 0,
            "max_iters": 1_000_000,
            "mtime": 0,
            "size_tier": str(ov.get("size_tier", "?")),
            "n_compartments": _to_int(ov.get("n_compartments", 0)),
            "ratio_mode": str(ov.get("translation_ratio_mode", "?")),
            "translation_ratio": _to_float(ov.get("translation_ratio", 0)),
            "locked": (lock_dir / f"{rid}.lock").exists(),
        })
        seen.add(rid)

    # Fill in unclaimed (no run dir yet) from grid overrides
    n_missing = len(grid_overrides) - len(configs)
    for i in range(max(0, n_missing)):
        ov = grid_overrides[i] if i < len(grid_overrides) else {}
        configs.append({
            "rid": f"_unclaimed_{i}",
            "iter_num": 0,
            "max_iters": 1_000_000,
            "mtime": 0,
            "size_tier": str(ov.get("model.size_tier", "?")),
            "n_compartments": ov.get("experiment.n_compartments", 0),
            "ratio_mode": str(ov.get("experiment.translation_ratio_mode", "?")),
            "translation_ratio": ov.get("experiment.translation_ratio", 0),
            "locked": False,
        })

    heartbeats = read_heartbeats(project)
    return configs, rid_to_job, n_run, n_pend, heartbeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep", help="Path to sweep YAML")
    parser.add_argument("--refresh", type=int, default=15)
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    parser.add_argument("--hide-done", action="store_true", help="Hide finished runs from grid")
    parser.add_argument("--no-workers", action="store_true", help="Hide workers panel")
    args = parser.parse_args()

    project, group, params = parse_sweep(args.sweep)
    grid_overrides = expand_grid(params)
    console = Console()
    eta_tracker = ETATracker()

    if args.once:
        configs, rid_to_job, n_run, n_pend, heartbeats = collect(
            project, group, grid_overrides, eta_tracker
        )
        hb = None if args.no_workers else heartbeats
        panel = Panel(
            build(group, configs, rid_to_job, eta_tracker, n_run, n_pend,
                  hide_done=args.hide_done, heartbeats=hb),
            title="Sweep Dashboard", border_style="cyan",
        )
        console.print(panel)
        return

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            try:
                configs, rid_to_job, n_run, n_pend, heartbeats = collect(
                    project, group, grid_overrides, eta_tracker
                )
                hb = None if args.no_workers else heartbeats
                live.update(Panel(
                    build(group, configs, rid_to_job, eta_tracker, n_run, n_pend,
                          hide_done=args.hide_done, heartbeats=hb),
                    title="Sweep Dashboard", border_style="cyan",
                ))
                time.sleep(args.refresh)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
