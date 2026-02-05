from dataclasses import asdict, dataclass, field
from typing import Literal

from tyro.conf import FlagConversionOff


@dataclass(frozen=True)
class Job:
    """Top-level job options and metadata."""

    # Optional path to a TOML config file (also used by ConfigManager for preloading)
    config_file: str | None = None


@dataclass(frozen=True)
class Data:
    source: Literal["pretokenized", "uniform"] = "pretokenized"
    train_bin: str = ""
    val_bin: str | None = None
    uniform_seed: int = 0


@dataclass(frozen=True)
class Model:
    # Match defaults from GPTConfig / train.py
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = False
    weight_tying: bool = True
    # Optional: preset size tier (e.g., "8-32", "8-64", "8-128", "8-256")
    size_tier: str | None = None
    # Optional advanced options (set programmatically in train.py)
    # When shared_token_embeddings is enabled, embedding_vocab_size should be base_vocab_size + 1.
    embedding_vocab_size: int | None = None
    shared_token_embeddings: bool = False
    use_compartment_embeddings: bool = False
    # These are provided so the model has the necessary context when advanced options are used.
    base_vocab_size: int | None = None
    max_compartments: int | None = None
    translation_token_id: int | None = None
    # If true (and not using shared_token_embeddings), clone base compartment
    # token embeddings across all compartments at initialization time.
    copy_compartment_embeddings: bool = False
    # If true, clone the lm_head rows for the base vocab across compartments.
    copy_compartment_lm_head: bool = False
    # vocab_size is derived from dataset meta by default
    vocab_size: int | None = None


@dataclass(frozen=True)
class Init:
    # 'scratch' | 'resume' | 'gpt2' | 'gpt2-medium' | 'gpt2-large' | 'gpt2-xl'
    init_from: str = "scratch"


@dataclass(frozen=True)
class Optimizer:
    learning_rate: float = 5e-2
    weight_decay: float = 0
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0


@dataclass(frozen=True)
class LRScheduler:
    # decay_lr: bool = True
    warmup_iters: int = 1000
    # lr_decay_iters: int = 600000
    # min_lr: float = 6e-5


@dataclass(frozen=True)
class Training:
    max_iters: int = 600000
    gradient_accumulation_steps: int = 5 * 8
    batch_size: int = 12
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    seed: int = 1024
    always_save_checkpoint: bool = True


@dataclass(frozen=True)
class Distributed:
    backend: str = "nccl"  # 'nccl', 'gloo', etc.


@dataclass(frozen=True)
class System:
    device: str = "cuda"  # 'cpu', 'cuda', 'cuda:0', ...
    # 'auto' picks bfloat16 if supported, else float16; can be 'float32'|'bfloat16'|'float16'
    dtype: str = "auto"
    compile: bool = True


@dataclass(frozen=True)
class Logging:
    wandb_log: bool = False
    wandb_project: str = "owt"
    wandb_run_name: str = "gpt2"
    wandb_group: str | None = None
    wandb_notes: str | None = None
    # Folders; manager will ensure they exist
    log_folder: str = "out"
    checkpoint_folder: str = "out"
    # If true, buffer wandb log calls and only flush after a checkpoint is saved.
    # Use this on preemptible/time-limited Slurm jobs to keep wandb state in sync
    # with checkpoint state.
    wandb_buffer: bool = False


# @dataclass(frozen=True)
# class Experiment:
#     """Experiment-specific options for assignment generation."""
#     # Mapping from compartment id (e.g., "0") or translation (e.g., "0>1") to weight
#     weights: dict[str, float] = field(default_factory=dict)
#     # Shuffle seed for deterministic ordering (defaults to 0; you can override in TOML)
#     assignment_seed: int = 0
#     # Maximum number of compartments. REQUIRED: must be provided in config.
#     max_compartments: int | None = None
#     # Advanced options
#     # If true, use one shared token embedding table of size base_vocab+1 and map inputs modulo base_vocab
#     shared_token_embeddings: bool = False
#     # If true, add a learned compartment embedding (max_compartments x n_embd) to token+pos embeddings
#     use_compartment_embeddings: bool = False
#     # If true and not using shared_token_embeddings, clone base token embeddings
#     # across compartments during initialization (model-side behavior).
#     copy_compartment_embeddings: bool = False
#     copy_compartment_lm_head: bool = False
#     # If true, use per-compartment permutations of base tokens. Model/tokenizer
#     # vocab becomes base_vocab+1 (translation token only) and tokens are mapped
#     # through a seeded permutation per compartment at data loading time.
#     permute_tokens_per_compartment: bool = False


@dataclass(frozen=True)
class Experiment:
    """Experiment-specific options for assignment generation."""

    # n
    n_compartments: int = 2
    # Whether we're in experiments 1,3 or 2,4
    compartment_scaling: Literal["equal", "unequal"] = "equal"
    # Scaling factor for translation tokens; 0 = no translations, 1 = as much
    # translation data as any one domain
    translation_ratio: float = 0
    # How to interpret translation_ratio:
    # - "compartment": 1 = as much translation data as any one compartment (default)
    # - "absolute": ratio of overall data that is translation; 1 = all translation data
    translation_ratio_mode: Literal["compartment", "absolute"] = "compartment"
    # Shuffle seed for deterministic ordering
    assignment_seed: int = 0
    # Maximum number of compartments. REQUIRED: must be provided in config.
    max_compartments: int | None = None
    # Advanced options
    # If true, use one shared token embedding table of size base_vocab+1 and map inputs
    # modulo base_vocab
    shared_token_embeddings: FlagConversionOff[bool] = False
    # If true, add a learned compartment embedding (max_compartments x n_embd) to
    # token+pos embeddings
    use_compartment_embeddings: FlagConversionOff[bool] = False
    # If true and not using shared_token_embeddings, clone base token embeddings
    # across compartments during initialization (model-side behavior).
    copy_compartment_embeddings: FlagConversionOff[bool] = False
    copy_compartment_lm_head: FlagConversionOff[bool] = False
    # If true, use per-compartment permutations of base tokens. Model/tokenizer
    # vocab becomes base_vocab+1 (translation token only) and tokens are mapped
    # through a seeded permutation per compartment at data loading time.
    permute_tokens_per_compartment: FlagConversionOff[bool] = True
    # When permuting tokens per compartment, controls whether model *inputs* are
    # also permuted. If False, inputs use the unpermuted base tokens while
    # targets remain in the permuted id space.
    permute_input_tokens_per_compartment: FlagConversionOff[bool] = True


@dataclass(frozen=True)
class JobConfig:
    """Configuration container for training."""

    job: Job = field(default_factory=Job)
    data: Data = field(default_factory=Data)
    model: Model = field(default_factory=Model)
    init: Init = field(default_factory=Init)
    optimizer: Optimizer = field(default_factory=Optimizer)
    lr: LRScheduler = field(default_factory=LRScheduler)
    training: Training = field(default_factory=Training)
    distributed: Distributed = field(default_factory=Distributed)
    system: System = field(default_factory=System)
    logging: Logging = field(default_factory=Logging)
    experiment: Experiment = field(default_factory=Experiment)

    def to_dict(self) -> dict[str, any]:  # pyright: ignore
        return asdict(self)
