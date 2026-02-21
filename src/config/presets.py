from dataclasses import replace

from .job_config import JobConfig


# Predefined size tiers. d_head is fixed at 32; n_head is derived as n_embd // 32.
TIER_SPECS: dict[str, dict[str, int]] = {
    "8-32": {"n_layer": 8, "n_embd": 32, "batch_size": 512, "grad_accum": 4},
    "8-64": {"n_layer": 8, "n_embd": 64, "batch_size": 512, "grad_accum": 4},
    "8-128": {"n_layer": 8, "n_embd": 128, "batch_size": 512, "grad_accum": 4},
    "8-256": {"n_layer": 8, "n_embd": 256, "batch_size": 512, "grad_accum": 4},
}

# Batch/grad_accum presets for vocab_size=16384 with no permutation (compartment embedding mode).
# Tested on 80GB GPUs. Key is (n_embd, max_compartments) -> (batch_size, grad_accum).
# Effective batch size is always 2048.
BPE16384_BATCH_SPECS: dict[tuple[int, int], tuple[int, int]] = {
    # 8-32, 8-64, 8-128 all have the same requirements
    (32, 2): (2048, 1),
    (32, 3): (2048, 1),
    (32, 4): (1024, 2),
    (32, 5): (1024, 2),
    (32, 8): (512, 4),
    (32, 16): (256, 8),
    (64, 2): (2048, 1),
    (64, 3): (2048, 1),
    (64, 4): (1024, 2),
    (64, 5): (1024, 2),
    (64, 8): (512, 4),
    (64, 16): (256, 8),
    (128, 2): (2048, 1),
    (128, 3): (2048, 1),
    (128, 4): (1024, 2),
    (128, 5): (1024, 2),
    (128, 8): (512, 4),
    (128, 16): (256, 8),
    # 8-256 needs more accumulation
    (256, 2): (1024, 2),
    (256, 3): (2048, 1),
    (256, 4): (512, 4),
    (256, 5): (1024, 2),
    (256, 8): (256, 8),
    (256, 16): (128, 16),
}


BASELINE_VRAM_GB = 80.0


def scale_batch_for_vram(cfg: JobConfig, vram_bytes: int) -> JobConfig:
    """Scale batch_size up and grad_accum down based on available VRAM.

    Preserves effective batch size (batch_size * grad_accum). Only applies
    when current batch config was set by BPE16384_BATCH_SPECS.
    """
    vram_gb = vram_bytes / (1024**3)
    if vram_gb <= BASELINE_VRAM_GB:
        return cfg

    batch_size = cfg.training.batch_size
    grad_accum = cfg.training.gradient_accumulation_steps
    effective_bs = batch_size * grad_accum

    if grad_accum <= 1:
        return cfg  # already minimal accumulation

    max_batch = int(batch_size * vram_gb / BASELINE_VRAM_GB)
    new_batch = batch_size
    candidate = batch_size * 2
    while candidate <= max_batch and effective_bs % candidate == 0:
        new_batch = candidate
        candidate *= 2
    new_accum = effective_bs // new_batch

    if new_batch == batch_size:
        return cfg

    new_training = replace(
        cfg.training,
        batch_size=new_batch,
        gradient_accumulation_steps=new_accum,
    )
    return replace(cfg, training=new_training)


def apply_bpe16384_batch_config(cfg: JobConfig) -> JobConfig:
    """Auto-configure batch_size and gradient_accumulation_steps for vocab_size=16384.

    Only applies when:
    - vocab_size == 16384 (base vocab for bpe16384 tokenizer)
    - permute_tokens_per_compartment is False (compartment embedding mode)
    - n_embd and n_compartments match a known configuration

    Effective batch size is always 2048.
    """
    # Check preconditions
    if cfg.model.vocab_size != 16384:
        return cfg
    if cfg.experiment.permute_tokens_per_compartment:
        return cfg

    n_embd = cfg.model.n_embd
    n_comp = cfg.experiment.n_compartments
    if n_embd is None or n_comp is None:
        return cfg

    key = (n_embd, n_comp)
    if key not in BPE16384_BATCH_SPECS:
        return cfg

    batch_size, grad_accum = BPE16384_BATCH_SPECS[key]
    new_training = replace(
        cfg.training,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
    )
    return replace(cfg, training=new_training)


def apply_size_tier(cfg: JobConfig) -> JobConfig:
    """Apply a predefined size tier to the immutable JobConfig.

    Sets model.n_layer, model.n_embd, derives model.n_head with d_head fixed at 32,
    and updates training.batch_size and training.gradient_accumulation_steps.
    """
    tier = cfg.model.size_tier
    if not tier:
        return cfg
    if tier not in TIER_SPECS:
        raise KeyError(f"Unknown size_tier: {tier}")
    spec = TIER_SPECS[tier]
    n_embd = spec["n_embd"]
    if n_embd % 32 != 0:
        raise ValueError("n_embd must be divisible by 32 to keep d_head fixed at 32")
    n_head = n_embd // 32
    new_model = replace(
        cfg.model, n_layer=spec["n_layer"], n_embd=n_embd, n_head=n_head
    )
    new_training = replace(
        cfg.training,
        batch_size=spec["batch_size"],
        gradient_accumulation_steps=spec["grad_accum"],
    )
    return replace(cfg, model=new_model, training=new_training)
