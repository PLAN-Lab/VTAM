# SPDX-License-Identifier: Apache-2.0
# Original VTAM code. Copyright 2026 The VTAM Authors.
"""Shared helpers for the relative_eef_rot6d action mode.

Single source of truth used by BOTH ``libero_dataset.get_batch`` and the offline
``calculate_statistics.py`` pass so the relative action they build can never
drift, and by BOTH ``ge_trainer`` and ``custom_pipeline`` for the state-
conditioning zero-pad so train/infer input assembly stays identical.
"""

from __future__ import annotations

import numpy as np

from data.utils.pose_math import relativize_pose6_to_rot6d


def build_relative_eef_rot6d_action(action_raw, state_raw, indexes, n_previous, arm_dim=6):
    """Build the 13D relative-eef action target in RAW physical space.

    action_raw: (N, >=arm_dim+...) absolute action rows [pose6, gripper, force, ...].
    state_raw:  (N, >=arm_dim)     absolute state rows  [pose6, gripper, ...].
    indexes:    the SAME index array get_batch uses (length n_previous+action_chunk).
    returns:    (len(indexes), 13) = [rel_xyz(3), rel_rot6d(6), gripper(1), force(3)]

    The arm 6D is relativized against base6 = the last OBSERVED state before the
    chunk (row n_previous-1) via true SE(3) body-frame transform; gripper/force
    stay absolute (untouched).

    NOTE on the meaning of "action": in this dataset the raw source (hanoi zarr) is
    OBSERVATION-ONLY -- there is no recorded controller command. The converter copies
    the single observed [pose6, gripper] stream into BOTH `state` and `action[:7]`
    (only `action` additionally carries the derived 3D tactile-flow force). So here
    `action[:, :arm_dim]` == the OBSERVED future EEF poses, not a teleop/controller
    target. Interpret this 13D vector as a "future EEF trajectory target expressed
    relative to the current pose" (behavior cloning from observed poses), NOT as a
    real controller command from the original collection.
    """
    sel_action = np.asarray(action_raw, dtype=np.float32)[indexes]
    sel_state = np.asarray(state_raw, dtype=np.float32)[indexes]
    base6 = sel_state[n_previous - 1, :arm_dim]                     # (arm_dim,)
    rel_arm = relativize_pose6_to_rot6d(base6, sel_action[:, :arm_dim])  # (T, 9)
    return np.concatenate([rel_arm, sel_action[:, arm_dim:]], axis=1).astype(np.float32)  # (T, 13)


def pad_action_for_state_conditioning(action13, state_dim=10):
    """Zero-pad the diffused action to the shared action_proj_in width.

    action13: (..., action_out_channels) torch tensor -- the diffusion sample.
    returns:  (..., action_out_channels + state_dim) with the trailing state slot
              zeroed. This is FORWARD-INPUT ONLY; the diffusion state, target and
              scheduler step all stay at action13's width. Used identically by
              ge_trainer.py and custom_pipeline.py to avoid train/infer drift.
    """
    import torch

    zeros = torch.zeros(
        *action13.shape[:-1], state_dim, device=action13.device, dtype=action13.dtype
    )
    return torch.cat([action13, zeros], dim=-1)
