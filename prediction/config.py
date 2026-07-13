# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import asdict, dataclass


@dataclass
class HiVTConfig:
    """Configuration for the HiVT trajectory prediction model.

    Two standard variants are provided as factory methods:
      - HiVTConfig.hivt_64()  → HiVT-64  (embed_dim=64)
      - HiVTConfig.hivt_128() → HiVT-128 (embed_dim=128)
    """

    # Temporal dimensions
    historical_steps: int = 20
    future_steps: int = 30

    # Prediction heads
    num_modes: int = 6

    # Input feature dimensions
    node_dim: int = 2
    edge_dim: int = 2

    # Model capacity
    embed_dim: int = 64

    # Attention
    num_heads: int = 8
    dropout: float = 0.1

    # Encoder depth
    num_temporal_layers: int = 4
    num_global_layers: int = 3

    # Spatial context
    local_radius: float = 50.0

    # Coordinate rotation normalisation
    rotate: bool = True

    # Process all time-steps in a single batched forward pass
    parallel: bool = False

    # Optimiser
    lr: float = 5e-4
    weight_decay: float = 1e-4
    T_max: int = 64

    # ------------------------------------------------------------------ #
    # Preset factory methods                                               #
    # ------------------------------------------------------------------ #

    @classmethod
    def hivt_64(cls) -> 'HiVTConfig':
        """Return the default HiVT-64 configuration (embed_dim=64)."""
        return cls(embed_dim=64)

    @classmethod
    def hivt_128(cls) -> 'HiVTConfig':
        """Return the HiVT-128 configuration (embed_dim=128)."""
        return cls(embed_dim=128)

    def to_dict(self) -> dict:
        """Return a plain dictionary of all configuration values."""
        return asdict(self)
