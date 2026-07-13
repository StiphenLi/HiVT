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
from typing import Optional, Tuple

import torch

from models.hivt import HiVT
from prediction.config import HiVTConfig
from utils import TemporalData


class HiVTPredictor:
    """Drop-in prediction wrapper for the HiVT model.

    This class provides a clean, model-agnostic interface for trajectory
    prediction so that HiVT can be used as a direct replacement for any
    previous prediction module.

    Example usage
    -------------
    >>> config = HiVTConfig.hivt_64()
    >>> predictor = HiVTPredictor(config)
    >>> predictions, scores = predictor.predict(data)

    >>> # Load a pretrained checkpoint
    >>> predictor = HiVTPredictor.from_checkpoint("checkpoints/HiVT-64.ckpt")
    >>> predictions, scores = predictor.predict(data)
    """

    def __init__(self, config: Optional[HiVTConfig] = None) -> None:
        if config is None:
            config = HiVTConfig.hivt_64()
        self.config = config
        self.model = HiVT(**config.to_dict())

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str,
                        parallel: bool = False) -> 'HiVTPredictor':
        """Instantiate a predictor from a PyTorch-Lightning checkpoint.

        Parameters
        ----------
        checkpoint_path:
            Path to a ``.ckpt`` file saved by the Lightning trainer.
        parallel:
            Override the ``parallel`` flag stored in the checkpoint.
            Set to ``True`` when running multi-GPU inference.
        """
        model = HiVT.load_from_checkpoint(
            checkpoint_path=checkpoint_path, parallel=parallel)
        predictor = cls.__new__(cls)
        predictor.model = model
        predictor.config = HiVTConfig(**{
            k: v for k, v in model.hparams.items()
            if k in HiVTConfig.__dataclass_fields__
        })
        return predictor

    # ------------------------------------------------------------------
    # Core prediction API
    # ------------------------------------------------------------------

    def predict(
        self,
        data: TemporalData,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference and return multi-modal trajectory predictions.

        Parameters
        ----------
        data:
            A :class:`~utils.TemporalData` object (or a batched version
            produced by PyG's ``DataLoader``).

        Returns
        -------
        predictions : torch.Tensor, shape ``[num_modes, num_agents, future_steps, 2]``
            Predicted ``(x, y)`` positions for every mode and every agent
            in the scene, relative to the agent's position at the last
            observed time step.
        scores : torch.Tensor, shape ``[num_agents, num_modes]``
            Unnormalised log-probabilities (logits) for each mode.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        data = data.to(device)

        with torch.no_grad():
            y_hat, pi = self.model(data)

        # y_hat: [num_modes, N, H, 4]  (loc + scale when uncertain=True)
        # pi:    [N, num_modes]
        # Return only the location component (first 2 channels).
        return y_hat[..., :2], pi

    def predict_agent(
        self,
        data: TemporalData,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return multi-modal predictions for the focal agent only.

        Parameters
        ----------
        data:
            A :class:`~utils.TemporalData` batch.

        Returns
        -------
        predictions : torch.Tensor, shape ``[num_graphs, num_modes, future_steps, 2]``
            Predicted trajectories for the focal agent in each scene of
            the batch.
        scores : torch.Tensor, shape ``[num_graphs, num_modes]``
            Unnormalised mode scores for the focal agent.
        """
        y_hat, pi = self.predict(data)
        # y_hat: [num_modes, N, H, 2], pi: [N, num_modes]
        agent_index = data['agent_index']
        # Extract the focal agent: [num_modes, num_graphs, H, 2] → [num_graphs, num_modes, H, 2]
        y_hat_agent = y_hat[:, agent_index, :, :].permute(1, 0, 2, 3)
        pi_agent = pi[agent_index]
        return y_hat_agent, pi_agent

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> 'HiVTPredictor':
        """Move the underlying model to *device* and return ``self``."""
        self.model = self.model.to(device)
        return self

    def eval(self) -> 'HiVTPredictor':
        """Set the underlying model to evaluation mode and return ``self``."""
        self.model.eval()
        return self

    def train(self) -> 'HiVTPredictor':
        """Set the underlying model to training mode and return ``self``."""
        self.model.train()
        return self

    @property
    def local_radius(self) -> float:
        """The spatial neighbourhood radius used during data pre-processing."""
        return self.config.local_radius
