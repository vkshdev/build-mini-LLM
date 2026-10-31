from dataclasses import dataclass, field
from typing import Optional

try:
    import torch
    _DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _DEFAULT_DTYPE = "float16" if torch.cuda.is_available() else "float32"
except ImportError:
    _DEFAULT_DEVICE = "cpu"
    _DEFAULT_DTYPE = "float32"


@dataclass
class ModelConfig:
    """Configuration for the model."""
    vocab_size: int = 50260
    block_size: int = 256            # context window length
    n_layer: int = 8                 # transformer blocks
    n_head: int = 8                  # attention heads (d_k = 64)
    n_embd: int = 512                # embedding dimension (~28.3M params total)
    dropout: float = 0.1
    bias: bool = False               # False = faster, LLaMA style


@dataclass
class TrainingConfig:
    batch_size: int = 32
    gradient_accumulation_steps: int = 2  # effective batch size = 64
    learning_rate: float = 6e-4
    min_lr: float = 6e-5             # 10% of peak lr
    warmup_steps: int = 250
    max_steps: int = 10000           # ~160M tokens
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    eval_interval: int = 500
    eval_iters: int = 40
    log_interval: int = 50
    sample_interval: int = 500
    save_interval: int = 1000

    checkpoint_dir: str = "checkpoints"
    resume: bool = False
    resume_checkpoint_path: Optional[str] = None

    device: str = _DEFAULT_DEVICE
    dtype: str = _DEFAULT_DTYPE


@dataclass
class SFTConfig:
    learning_rate: float = 3e-5      # lower lr to avoid catastrophic forgetting
    min_lr: float = 3e-6
    max_steps: int = 1500            # ~35 mins on Colab T4
    warmup_steps: int = 100
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    eval_interval: int = 150
    checkpoint_path: str = "checkpoints/best_pretrained_28m.pt"


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "custom-llm-from-scratch"
    run_name: str = "tinystories-28m-pretrain"
    entity: Optional[str] = None
    notes: str = "28M Transformer on TinyStories"
    tags: list = field(default_factory=lambda: ["pretraining", "28M", "tinystories", "colab-t4"])
