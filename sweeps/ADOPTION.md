# Sweep Runner Adoption Guide

`sweep_runner.py` replaces `wandb agent` for running grid sweeps. It uses the same YAML files but handles coordination itself via file locks + wandb reads.

## What changes in your Slurm scripts

Wherever you had:

```bash
wandb agent pccl/translation-compression/<sweep-id>
```

Replace with:

```bash
python sweep_runner.py \
    --sweep sweeps/<your-sweep>.yaml \
    --cluster slurm \
    --wandb-buffer
```

That's it. Everything else (the wandb proxy setup, CUDA_VISIBLE_DEVICES, conda/venv activation, etc.) stays the same.

## Example: converting an existing Slurm script

**Before:**
```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=4:00:00

# ... proxy setup, venv activation, etc. ...

wandb agent pccl/translation-compression/cn8yurxt
```

**After:**
```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=4:00:00

# ... proxy setup, venv activation, etc. ...

export TC_STORAGE_ROOT="/nobackup/archive/grp/grp_pccl/vin/dev/translation-compression"

python sweep_runner.py \
    --sweep sweeps/bpe16384-n3-n5.yaml \
    --cluster slurm \
    --wandb-buffer
```

Submit one job per GPU:
```bash
sbatch --array=0-7 run_sweep.sh
```

## Key differences from `wandb agent`

| | `wandb agent` | `sweep_runner.py` |
|---|---|---|
| Config source | wandb server (sweep ID) | Local YAML file |
| Coordination | wandb server | File locks (intra-cluster) + wandb reads (cross-cluster) |
| Resume | Manual | Automatic — detects local checkpoints |
| Run IDs | Random from wandb | Deterministic from config hash |
| Output dirs | Timestamp-based | Deterministic: `out/{project}/{group}/{hash}_s{seed}/` |

## Environment variable: `TC_STORAGE_ROOT`

The storage root differs per cluster. Set this in your Slurm script or `.bashrc`:

```bash
# Slurm cluster
export TC_STORAGE_ROOT="/nobackup/archive/grp/grp_pccl/vin/dev/translation-compression"

# Local cluster (default, no need to set)
# TC_STORAGE_ROOT="/mnt/pccfs2/backed_up/vin/dev/translation-compression"
```

## CLI flags

| Flag | Purpose |
|---|---|
| `--sweep PATH` | Path to sweep YAML (required) |
| `--cluster NAME` | Cluster name stored in wandb config (default: `local`) |
| `--wandb-buffer` | Buffer wandb logs until checkpoint — use on preemptible jobs |
| `--status` | Print status table and exit (no GPU needed) |
| `--allow-behind` | Skip the git-behind-remote check |

## How it works

1. Parses the sweep YAML and expands the parameter grid
2. Builds each config (same pipeline as `train.py`: TOML + overrides + presets + seed mirroring)
3. Queries wandb for existing run states (single API call)
4. Picks a config: resumes local partial checkpoints first, then claims new work
5. Claims via atomic file lock (`out/{project}/.locks/{run_id}.lock`)
6. Calls `train.main()` directly (no subprocess)
7. When training finishes, loops back to step 3

On preemption (SIGKILL → Slurm requeue), the runner restarts, finds the checkpoint, and resumes.

## Checking status

No GPU needed:
```bash
python sweep_runner.py --sweep sweeps/bpe16384-n3-n5.yaml --status
```

## Running on both clusters simultaneously

Just run the same command on each cluster with different `--cluster` values. The runner checks wandb before claiming, so both clusters see each other's work and won't duplicate.
