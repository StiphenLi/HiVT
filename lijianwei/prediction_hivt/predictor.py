"""
HiVT-based trajectory prediction module (prediction_hivt).

This module provides a HiVT (Hierarchical Vector Transformer) predictor
that exposes the same public interface as the QCNet-based ``prediction``
module, allowing a drop-in switch between the two backends.

Typical usage::

    from lijianwei.prediction_hivt import HiVTPredictionModule
    from lijianwei.prediction_hivt.config import PredictionConfig

    cfg = PredictionConfig()
    cfg.inference.checkpoint_path = 'checkpoints/HiVT-128/checkpoints/epoch=63-step=411903.ckpt'
    cfg.inference.device = 'cuda'

    predictor = HiVTPredictionModule(cfg)

    # From a pre-processed TemporalData object
    result = predictor.predict(data)

    # From a raw Argoverse-style DataFrame
    result = predictor.predict_from_df(df)

    # From raw numpy position arrays
    result = predictor.predict_from_arrays(positions, object_types, city)

Each call returns a dict::

    {
        'trajectories':   np.ndarray  # [num_modes, future_steps, 2]   (x, y)
        'probabilities':  np.ndarray  # [num_modes]
        'uncertainties':  np.ndarray  # [num_modes, future_steps, 2]  (only when enabled)
    }
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

# ---------------------------------------------------------------------------
# Ensure the repository root is on sys.path so that top-level packages
# (models, datasets, utils, …) are importable regardless of cwd.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.hivt import HiVT  # noqa: E402
from utils import TemporalData  # noqa: E402

from .config.prediction_config_040702 import PredictionConfig
from .data_utils import arrays_to_temporal_data, df_to_temporal_data


class HiVTPredictionModule:
    """
    HiVT predictor with the same public interface as the QCNet predictor.

    Parameters
    ----------
    config:
        ``PredictionConfig`` instance.  Defaults to HiVT-128 settings.
        See ``prediction_hivt.config.prediction_config_040702`` for all
        available options.
    """

    def __init__(self, config: Optional[PredictionConfig] = None) -> None:
        if config is None:
            config = PredictionConfig()
        self.config = config

        self._device = torch.device(config.inference.device)
        self._model: Optional[HiVT] = None

        if config.inference.checkpoint_path is not None:
            self._load_checkpoint(config.inference.checkpoint_path)

    # ------------------------------------------------------------------
    # Public interface (mirrors QCNet predictor)
    # ------------------------------------------------------------------

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load (or hot-swap) a HiVT checkpoint at runtime."""
        self._load_checkpoint(checkpoint_path)

    def predict(
        self,
        data: Union[TemporalData, Data],
    ) -> Dict[str, np.ndarray]:
        """
        Run HiVT inference on a single pre-processed scene graph.

        Parameters
        ----------
        data:
            A ``TemporalData`` / PyG ``Data`` object as produced by
            ``data_utils.df_to_temporal_data`` or the Argoverse dataset.

        Returns
        -------
        dict with keys:

        ``'trajectories'`` — ``np.ndarray`` shape ``(num_modes, future_steps, 2)``
            Predicted (x, y) positions for the focal agent (AGENT) in the
            coordinate frame centred on AV at the last historical step.

        ``'probabilities'`` — ``np.ndarray`` shape ``(num_modes,)``
            Softmax mode probabilities.

        ``'uncertainties'`` — ``np.ndarray`` shape ``(num_modes, future_steps, 2)``
            Laplace scale parameters.  Empty array when
            ``config.inference.return_uncertainties`` is ``False``.
        """
        model = self._require_model()
        model.eval()

        data = data.to(self._device)

        with torch.no_grad():
            y_hat, pi = model(data)  # [F, N, future_steps, 4], [N, F]

        agent_index = int(data['agent_index'])

        # y_hat[..., :2] = predicted (x, y); y_hat[..., 2:] = Laplace scales
        trajectories = y_hat[:, agent_index, :, :2].cpu().numpy()   # [F, future_steps, 2]
        if self.config.inference.return_uncertainties:
            uncertainties = y_hat[:, agent_index, :, 2:].cpu().numpy()  # [F, future_steps, 2]
        else:
            uncertainties = np.empty(0)

        probs = F.softmax(pi[agent_index], dim=-1).cpu().numpy()  # [F]

        return {
            'trajectories': trajectories,
            'probabilities': probs,
            'uncertainties': uncertainties,
        }

    def predict_batch(
        self,
        data_list: List[Union[TemporalData, Data]],
    ) -> List[Dict[str, np.ndarray]]:
        """
        Run HiVT inference on a list of scene graphs.

        Parameters
        ----------
        data_list:
            List of ``TemporalData`` objects (one per scenario).

        Returns
        -------
        List of prediction dicts (same format as :meth:`predict`).
        """
        model = self._require_model()
        model.eval()

        batch = Batch.from_data_list(data_list).to(self._device)

        with torch.no_grad():
            y_hat, pi = model(batch)  # [F, N_total, future_steps, 4], [N_total, F]

        results = []
        agent_indices = batch['agent_index']
        for i in range(len(data_list)):
            agent_global_idx = int(agent_indices[i])
            trajectories = y_hat[:, agent_global_idx, :, :2].cpu().numpy()
            if self.config.inference.return_uncertainties:
                uncertainties = y_hat[:, agent_global_idx, :, 2:].cpu().numpy()
            else:
                uncertainties = np.empty(0)
            probs = F.softmax(pi[agent_global_idx], dim=-1).cpu().numpy()
            results.append({
                'trajectories': trajectories,
                'probabilities': probs,
                'uncertainties': uncertainties,
            })
        return results

    def predict_from_df(
        self,
        df: pd.DataFrame,
        am=None,
    ) -> Dict[str, np.ndarray]:
        """
        Accept a raw Argoverse CSV DataFrame and return predictions.

        Parameters
        ----------
        df:
            Raw Argoverse scenario DataFrame (columns: TIMESTAMP, TRACK_ID,
            OBJECT_TYPE, X, Y, CITY_NAME).
        am:
            Optional ``ArgoverseMap`` instance for lane features.

        Returns
        -------
        Prediction dict — same as :meth:`predict`.
        """
        data = df_to_temporal_data(
            df,
            local_radius=self.config.model.local_radius,
            split='val',
            am=am,
        )
        return self.predict(data)

    def predict_from_arrays(
        self,
        positions: np.ndarray,
        object_types: List[str],
        city: str,
        timestamps: Optional[np.ndarray] = None,
        am=None,
    ) -> Dict[str, np.ndarray]:
        """
        Accept raw numpy position arrays and return predictions.

        Parameters
        ----------
        positions:
            Float array of shape ``(N, T, 2)`` with ``(x, y)`` world
            coordinates.  Missing observations should be ``np.nan``.
        object_types:
            List of ``N`` strings (``'AV'``, ``'AGENT'``, ``'OTHERS'``).
            Must contain exactly one ``'AV'`` entry.
        city:
            Argoverse city identifier (e.g. ``'MIA'``, ``'PIT'``).
        timestamps:
            Optional 1-D float array of length ``T``.
        am:
            Optional ``ArgoverseMap`` instance.

        Returns
        -------
        Prediction dict — same as :meth:`predict`.
        """
        data = arrays_to_temporal_data(
            positions,
            object_types,
            city,
            timestamps=timestamps,
            local_radius=self.config.model.local_radius,
            am=am,
        )
        return self.predict(data)

    # ------------------------------------------------------------------
    # Model construction helpers
    # ------------------------------------------------------------------

    def build_model(self) -> HiVT:
        """Instantiate a fresh (randomly-initialised) HiVT from the config."""
        model = HiVT(**self.config.to_hivt_kwargs())
        model.to(self._device)
        self._model = model
        return model

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f'HiVT checkpoint not found: {checkpoint_path}'
            )
        model = HiVT.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=self._device,
            parallel=self.config.model.parallel,
        )
        model.to(self._device)
        model.eval()
        self._model = model

    def _require_model(self) -> HiVT:
        if self._model is None:
            raise RuntimeError(
                'No model loaded.  Pass a checkpoint_path in InferenceConfig '
                'or call build_model() / load_checkpoint() before predict().'
            )
        return self._model
