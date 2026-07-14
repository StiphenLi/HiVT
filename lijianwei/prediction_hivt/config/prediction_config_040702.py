"""
Configuration for HiVT-based trajectory prediction module.

This configuration replaces the former QCNet-based prediction setup
and adapts HiVT (Hierarchical Vector Transformer) as the prediction model.

Model reference:
    HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction
    https://github.com/ZikangZhou/HiVT
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HiVTModelConfig:
    """HiVT model hyperparameters."""

    # Number of historical time steps (20 steps at 10 Hz = 2 s of history)
    historical_steps: int = 20

    # Number of future time steps to predict (30 steps at 10 Hz = 3 s)
    future_steps: int = 30

    # Number of prediction modes (multimodal output)
    num_modes: int = 6

    # Whether to rotate scene so that AV heading aligns with x-axis
    rotate: bool = True

    # Node feature dimensionality (x, y displacements)
    node_dim: int = 2

    # Edge feature dimensionality (relative x, y)
    edge_dim: int = 2

    # Transformer embedding dimension (128 = HiVT-128 variant)
    embed_dim: int = 128

    # Number of attention heads
    num_heads: int = 8

    # Dropout probability
    dropout: float = 0.1

    # Number of temporal transformer layers in local encoder
    num_temporal_layers: int = 4

    # Number of global interaction layers
    num_global_layers: int = 3

    # Local neighbourhood radius (metres) used to build the scene graph
    local_radius: float = 50.0

    # Process time steps in parallel inside the local encoder (faster on GPU)
    parallel: bool = False


@dataclass
class TrainingConfig:
    """Training hyperparameters (used when fine-tuning the model)."""

    # Initial learning rate
    lr: float = 5e-4

    # AdamW weight decay
    weight_decay: float = 1e-4

    # CosineAnnealingLR period (epochs)
    T_max: int = 64

    # Total training epochs
    max_epochs: int = 64

    # Training batch size
    train_batch_size: int = 32

    # Validation batch size
    val_batch_size: int = 32

    # Shuffle training data
    shuffle: bool = True

    # DataLoader worker count
    num_workers: int = 8

    # Pin memory for faster GPU transfer
    pin_memory: bool = True

    # Keep DataLoader workers alive between epochs
    persistent_workers: bool = True

    # GPU count
    gpus: int = 1

    # Metric used for checkpoint selection
    monitor: str = 'val_minFDE'

    # Number of best checkpoints to keep
    save_top_k: int = 5


@dataclass
class DataConfig:
    """Dataset and pre-processing settings."""

    # Root directory of the Argoverse v1 dataset
    dataset_root: str = '/data/argoverse'

    # Time frequency of the dataset (Hz)
    frequency_hz: float = 10.0

    # Radius (metres) for building lane–actor cross-attention edges
    local_radius: float = 50.0


@dataclass
class InferenceConfig:
    """Runtime / inference settings."""

    # Path to a pre-trained HiVT checkpoint (.ckpt)
    checkpoint_path: Optional[str] = None

    # Device to run inference on: 'cpu', 'cuda', 'cuda:0', …
    device: str = 'cpu'

    # Inference batch size
    batch_size: int = 32

    # DataLoader workers during inference
    num_workers: int = 4

    # Whether to return uncertainty estimates (scale parameters) alongside
    # predicted positions.  Set False to return x,y only.
    return_uncertainties: bool = False


@dataclass
class PredictionConfig:
    """
    Top-level configuration for the HiVT prediction module.

    Usage::

        from lijianwei.prediction.config import PredictionConfig

        cfg = PredictionConfig()
        cfg.model.embed_dim = 64          # switch to HiVT-64
        cfg.inference.checkpoint_path = 'checkpoints/HiVT-64/checkpoints/epoch=63-step=411903.ckpt'
    """

    model: HiVTModelConfig = field(default_factory=HiVTModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def to_hivt_kwargs(self) -> dict:
        """Return a flat dict of HiVT.__init__ keyword arguments."""
        m = self.model
        t = self.training
        return dict(
            historical_steps=m.historical_steps,
            future_steps=m.future_steps,
            num_modes=m.num_modes,
            rotate=m.rotate,
            node_dim=m.node_dim,
            edge_dim=m.edge_dim,
            embed_dim=m.embed_dim,
            num_heads=m.num_heads,
            dropout=m.dropout,
            num_temporal_layers=m.num_temporal_layers,
            num_global_layers=m.num_global_layers,
            local_radius=m.local_radius,
            parallel=m.parallel,
            lr=t.lr,
            weight_decay=t.weight_decay,
            T_max=t.T_max,
        )
