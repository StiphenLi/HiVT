"""
Data pre-processing utilities for the HiVT prediction module.

This module re-uses the core Argoverse v1 pre-processing logic from
``datasets/argoverse_v1_dataset.py`` and exposes helpers that convert
raw trajectory DataFrames (or plain numpy arrays) into ``TemporalData``
objects ready for the HiVT model.
"""
from itertools import permutations, product
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from utils import TemporalData


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def df_to_temporal_data(
    df: pd.DataFrame,
    local_radius: float = 50.0,
    split: str = 'val',
    am=None,
) -> TemporalData:
    """
    Convert a raw Argoverse v1 scenario CSV (loaded as a DataFrame) into a
    ``TemporalData`` object suitable for HiVT inference.

    Parameters
    ----------
    df:
        Raw scenario DataFrame with columns
        ``TIMESTAMP``, ``TRACK_ID``, ``OBJECT_TYPE``, ``X``, ``Y``,
        ``CITY_NAME``.
    local_radius:
        Neighbourhood radius in metres used to build the lane–actor graph.
        Must match the value used during model training (default 50 m).
    split:
        Dataset split; one of ``'train'``, ``'val'``, ``'test'``.
        When ``'test'`` the ``y`` (future trajectory) field is set to ``None``.
    am:
        Optional ``ArgoverseMap`` instance.  When ``None`` lane features are
        set to empty tensors (useful for environments where the map API is not
        available).

    Returns
    -------
    TemporalData
        A graph-structured scene representation accepted by HiVT.
    """
    kwargs = _process_scenario(split, df, am, local_radius)
    return TemporalData(**kwargs)


def arrays_to_temporal_data(
    positions: np.ndarray,
    object_types: List[str],
    city: str,
    timestamps: Optional[np.ndarray] = None,
    local_radius: float = 50.0,
    am=None,
) -> TemporalData:
    """
    Build a ``TemporalData`` object from raw numpy arrays.

    Parameters
    ----------
    positions:
        Float array of shape ``(N, T, 2)`` with ``(x, y)`` world coordinates
        for each of the ``N`` agents across ``T`` time steps.
        Missing observations should be filled with ``np.nan``.
    object_types:
        List of ``N`` strings (e.g. ``'AV'``, ``'AGENT'``, ``'OTHERS'``).
        The scene must contain exactly one ``'AV'`` entry.
    city:
        Argoverse city name (e.g. ``'MIA'``, ``'PIT'``).
    timestamps:
        Optional 1-D array of length ``T``.  If ``None`` integers
        ``0 … T-1`` are used.
    local_radius:
        Neighbourhood radius in metres (default 50 m).
    am:
        Optional ``ArgoverseMap`` instance.

    Returns
    -------
    TemporalData
    """
    n_agents, n_steps, _ = positions.shape
    if timestamps is None:
        timestamps = np.arange(n_steps, dtype=np.float64)

    rows = []
    for node_idx, (obj_type, traj) in enumerate(zip(object_types, positions)):
        track_id = f'agent_{node_idx}'
        for t_idx, t in enumerate(timestamps):
            x_val, y_val = traj[t_idx]
            if np.isnan(x_val) or np.isnan(y_val):
                continue
            rows.append({
                'TIMESTAMP': t,
                'TRACK_ID': track_id,
                'OBJECT_TYPE': obj_type,
                'X': x_val,
                'Y': y_val,
                'CITY_NAME': city,
            })

    df = pd.DataFrame(rows)
    historical_steps = 20
    all_ts = list(np.sort(df['TIMESTAMP'].unique()))
    split = 'val' if n_steps > historical_steps else 'test'
    return df_to_temporal_data(df, local_radius=local_radius, split=split, am=am)


# ---------------------------------------------------------------------------
# Internal helpers (adapted from datasets/argoverse_v1_dataset.py)
# ---------------------------------------------------------------------------

def _process_scenario(
    split: str,
    df: pd.DataFrame,
    am,
    radius: float,
) -> Dict:
    """Core scenario processing logic mirroring ``process_argoverse``."""

    timestamps = list(np.sort(df['TIMESTAMP'].unique()))
    historical_timestamps = timestamps[:20]
    historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]
    actor_ids = list(historical_df['TRACK_ID'].unique())
    df = df[df['TRACK_ID'].isin(actor_ids)]
    num_nodes = len(actor_ids)

    av_df = df[df['OBJECT_TYPE'] == 'AV'].iloc
    av_index = actor_ids.index(av_df[0]['TRACK_ID'])
    agent_df = df[df['OBJECT_TYPE'] == 'AGENT'].iloc
    agent_index = actor_ids.index(agent_df[0]['TRACK_ID'])
    city = df['CITY_NAME'].values[0]

    # Scene centred at AV at time step 19
    origin = torch.tensor([av_df[19]['X'], av_df[19]['Y']], dtype=torch.float)
    av_heading_vector = origin - torch.tensor([av_df[18]['X'], av_df[18]['Y']], dtype=torch.float)
    theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
    rotate_mat = torch.tensor([
        [torch.cos(theta), -torch.sin(theta)],
        [torch.sin(theta),  torch.cos(theta)],
    ])

    # Initialise tensors
    num_total_steps = len(timestamps)
    x = torch.zeros(num_nodes, num_total_steps, 2, dtype=torch.float)
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    padding_mask = torch.ones(num_nodes, num_total_steps, dtype=torch.bool)
    bos_mask = torch.zeros(num_nodes, 20, dtype=torch.bool)
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)

    for actor_id, actor_df in df.groupby('TRACK_ID'):
        node_idx = actor_ids.index(actor_id)
        node_steps = [timestamps.index(ts) for ts in actor_df['TIMESTAMP']]
        padding_mask[node_idx, node_steps] = False
        if padding_mask[node_idx, 19]:
            padding_mask[node_idx, 20:] = True
        xy = torch.from_numpy(
            np.stack([actor_df['X'].values, actor_df['Y'].values], axis=-1)
        ).float()
        x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
        node_historical_steps = [s for s in node_steps if s < 20]
        if len(node_historical_steps) > 1:
            heading_vector = (
                x[node_idx, node_historical_steps[-1]]
                - x[node_idx, node_historical_steps[-2]]
            )
            rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
        else:
            padding_mask[node_idx, 20:] = True

    bos_mask[:, 0] = ~padding_mask[:, 0]
    bos_mask[:, 1:20] = padding_mask[:, :19] & ~padding_mask[:, 1:20]

    positions = x.clone()
    x[:, 20:] = torch.where(
        (padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1),
        torch.zeros(num_nodes, num_total_steps - 20, 2),
        x[:, 20:] - x[:, 19].unsqueeze(-2),
    )
    x[:, 1:20] = torch.where(
        (padding_mask[:, :19] | padding_mask[:, 1:20]).unsqueeze(-1),
        torch.zeros(num_nodes, 19, 2),
        x[:, 1:20] - x[:, :19],
    )
    x[:, 0] = torch.zeros(num_nodes, 2)

    # Lane features
    if am is not None:
        df_19 = df[df['TIMESTAMP'] == timestamps[19]]
        node_inds_19 = [actor_ids.index(aid) for aid in df_19['TRACK_ID']]
        node_positions_19 = torch.from_numpy(
            np.stack([df_19['X'].values, df_19['Y'].values], axis=-1)
        ).float()
        (lane_vectors, is_intersections, turn_directions, traffic_controls,
         lane_actor_index, lane_actor_vectors) = _get_lane_features(
            am, node_inds_19, node_positions_19, origin, rotate_mat, city, radius
        )
    else:
        lane_vectors = torch.zeros(0, 2, dtype=torch.float)
        is_intersections = torch.zeros(0, dtype=torch.uint8)
        turn_directions = torch.zeros(0, dtype=torch.uint8)
        traffic_controls = torch.zeros(0, dtype=torch.uint8)
        lane_actor_index = torch.zeros(2, 0, dtype=torch.long)
        lane_actor_vectors = torch.zeros(0, 2, dtype=torch.float)

    y = None if split == 'test' else x[:, 20:]

    return {
        'x': x[:, :20],                     # [N, 20, 2]
        'positions': positions,              # [N, 50, 2]
        'edge_index': edge_index,            # [2, N*(N-1)]
        'y': y,                              # [N, 30, 2] or None
        'num_nodes': num_nodes,
        'padding_mask': padding_mask,        # [N, 50]
        'bos_mask': bos_mask,               # [N, 20]
        'rotate_angles': rotate_angles,      # [N]
        'lane_vectors': lane_vectors,        # [L, 2]
        'is_intersections': is_intersections,
        'turn_directions': turn_directions,
        'traffic_controls': traffic_controls,
        'lane_actor_index': lane_actor_index,
        'lane_actor_vectors': lane_actor_vectors,
        'seq_id': 0,
        'av_index': av_index,
        'agent_index': agent_index,
        'city': city,
        'origin': origin.unsqueeze(0),
        'theta': theta,
    }


def _get_lane_features(
    am,
    node_inds: List[int],
    node_positions: torch.Tensor,
    origin: torch.Tensor,
    rotate_mat: torch.Tensor,
    city: str,
    radius: float,
) -> Tuple[torch.Tensor, ...]:
    """Extract lane features for all actors at the current time step."""

    lane_positions, lane_vectors = [], []
    is_intersections, turn_directions, traffic_controls = [], [], []
    lane_ids: set = set()

    for node_position in node_positions:
        lane_ids.update(
            am.get_lane_ids_in_xy_bbox(node_position[0], node_position[1], city, radius)
        )

    node_positions = torch.matmul(node_positions - origin, rotate_mat).float()

    for lane_id in lane_ids:
        lane_centerline = torch.from_numpy(
            am.get_lane_segment_centerline(lane_id, city)[:, :2]
        ).float()
        lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)
        is_intersection = am.lane_is_in_intersection(lane_id, city)
        turn_direction = am.get_lane_turn_direction(lane_id, city)
        traffic_control = am.lane_has_traffic_control_measure(lane_id, city)
        lane_positions.append(lane_centerline[:-1])
        lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
        count = len(lane_centerline) - 1
        is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
        if turn_direction == 'NONE':
            td = 0
        elif turn_direction == 'LEFT':
            td = 1
        elif turn_direction == 'RIGHT':
            td = 2
        else:
            raise ValueError(f'Unknown turn direction: {turn_direction}')
        turn_directions.append(td * torch.ones(count, dtype=torch.uint8))
        traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))

    lane_positions = torch.cat(lane_positions, dim=0)
    lane_vectors = torch.cat(lane_vectors, dim=0)
    is_intersections = torch.cat(is_intersections, dim=0)
    turn_directions = torch.cat(turn_directions, dim=0)
    traffic_controls = torch.cat(traffic_controls, dim=0)

    lane_actor_index = torch.LongTensor(
        list(product(torch.arange(lane_vectors.size(0)), node_inds))
    ).t().contiguous()
    lane_actor_vectors = (
        lane_positions.repeat_interleave(len(node_inds), dim=0)
        - node_positions.repeat(lane_vectors.size(0), 1)
    )
    mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
    lane_actor_index = lane_actor_index[:, mask]
    lane_actor_vectors = lane_actor_vectors[mask]

    return (
        lane_vectors, is_intersections, turn_directions,
        traffic_controls, lane_actor_index, lane_actor_vectors,
    )
