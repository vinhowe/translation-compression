from dataclasses import replace

from .job_config import JobConfig


# Predefined size tiers. d_head is fixed at 32; n_head is derived as n_embd // 32.
TIER_SPECS: dict[str, dict[str, int]] = {
    "8-32": {"n_layer": 8, "n_embd": 32, "batch_size": 512, "grad_accum": 4},
    "8-64": {"n_layer": 8, "n_embd": 64, "batch_size": 512, "grad_accum": 4},
    "8-128": {"n_layer": 8, "n_embd": 128, "batch_size": 512, "grad_accum": 4},
    "8-256": {"n_layer": 8, "n_embd": 256, "batch_size": 512, "grad_accum": 4},
}


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
