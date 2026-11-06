# SPDX-License-Identifier: Apache-2.0
# Original VTAM code. Copyright 2026 The VTAM Authors.
"""Shared, vectorized SE(3) + rot6d pose-math helpers.

Single source of truth for the relative_eef_rot6d action mode so that the
dataset (``libero_dataset.get_batch``), the offline stats pass
(``calculate_statistics.py``) and deployment (``vtam_server.py`` /
``serve_vtam_policy.py``) all use *identical* math and rot6d conventions.

Conventions (pinned to match ``scripts/serve_vtam_policy.py``):
  * absolute pose6 = ``[xyz(3), rotvec(3)]`` (axis-angle, scipy ``Rotation``).
  * rot6d = the FIRST TWO COLUMNS of the rotation matrix, laid out as
    ``[col0(3), col1(3)]`` (i.e. ``swapaxes(R[..., :3, :2], -1, -2).reshape(6)``).
  * ``rot6d_to_mat`` rebuilds a proper rotation via Gram-Schmidt:
    ``b1 = norm(a1); b2 = norm(a2 - <b1,a2> b1); b3 = b1 x b2``,
    ``R = stack([b1, b2, b3], axis=-1)`` (columns).

All functions are batched over arbitrary leading dims unless noted.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.spatial.transform import Rotation as _Rotation
except Exception:  # pragma: no cover - scipy always present in the training env
    _Rotation = None


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, eps)


# --------------------------------------------------------------------------- #
# rot6d <-> rotation matrix (first-two-columns + Gram-Schmidt)
# --------------------------------------------------------------------------- #
def rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    """(..., 6) rot6d -> (..., 3, 3) proper rotation matrix (columns)."""
    d6 = np.asarray(d6, dtype=np.float32)
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-1)


def mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation -> (..., 6) rot6d = [col0(3), col1(3)]."""
    mat = np.asarray(mat, dtype=np.float32)
    return np.swapaxes(mat[..., :3, :2], -1, -2).reshape(mat.shape[:-2] + (6,))


# --------------------------------------------------------------------------- #
# pose6 (xyz + rotvec) <-> 4x4 homogeneous matrix  (batched)
# --------------------------------------------------------------------------- #
def _require_scipy() -> None:
    if _Rotation is None:
        raise RuntimeError("scipy is required for rotvec <-> matrix conversion")


def pose6_to_mat(pose6: np.ndarray) -> np.ndarray:
    """(..., 6) pose6=[xyz, rotvec] -> (..., 4, 4) homogeneous transform."""
    _require_scipy()
    pose6 = np.asarray(pose6, dtype=np.float32)
    lead = pose6.shape[:-1]
    flat = pose6.reshape(-1, 6)
    mats = np.broadcast_to(np.eye(4, dtype=np.float32), (flat.shape[0], 4, 4)).copy()
    mats[:, :3, 3] = flat[:, :3]
    mats[:, :3, :3] = _Rotation.from_rotvec(flat[:, 3:6]).as_matrix().astype(np.float32)
    return mats.reshape(lead + (4, 4))


def mat_to_pose6(mat: np.ndarray) -> np.ndarray:
    """(..., 4, 4) homogeneous transform -> (..., 6) pose6=[xyz, rotvec]."""
    _require_scipy()
    mat = np.asarray(mat, dtype=np.float32)
    lead = mat.shape[:-2]
    flat = mat.reshape(-1, 4, 4)
    pose = np.zeros((flat.shape[0], 6), dtype=np.float32)
    pose[:, :3] = flat[:, :3, 3]
    pose[:, 3:6] = _Rotation.from_matrix(flat[:, :3, :3]).as_rotvec().astype(np.float32)
    return pose.reshape(lead + (6,))


# --------------------------------------------------------------------------- #
# relative action (SE(3) body-frame) <-> absolute pose, rot6d representation
# --------------------------------------------------------------------------- #
def relativize_pose6_to_rot6d(base6: np.ndarray, targets6: np.ndarray) -> np.ndarray:
    """Body-frame relative action, rot6d.

    base6:    (6,) or (T, 6) current EEF pose6 (xyz + rotvec).
    targets6: (T, 6) future absolute EEF pose6.
    returns:  (T, 9) = [rel_xyz(3), rel_rot6d(6)] with T_rel = inv(T_base) @ T_target.
    """
    targets6 = np.asarray(targets6, dtype=np.float32)
    base6 = np.asarray(base6, dtype=np.float32)
    if base6.ndim == 1:
        base6 = np.broadcast_to(base6, targets6.shape)
    T_base = pose6_to_mat(base6)          # (T, 4, 4)
    T_targ = pose6_to_mat(targets6)       # (T, 4, 4)
    T_rel = np.linalg.inv(T_base) @ T_targ
    rel_xyz = T_rel[..., :3, 3]                    # (T, 3)
    rel_rot6d = mat_to_rot6d(T_rel[..., :3, :3])   # (T, 6)
    return np.concatenate([rel_xyz, rel_rot6d], axis=-1).astype(np.float32)


def compose_rot6d_to_pose6(base6: np.ndarray, rel9: np.ndarray) -> np.ndarray:
    """Deploy inverse: absolute pose6 from base pose + relative rot6d action.

    base6: (6,) or (T, 6) current EEF pose6.
    rel9:  (T, 9) = [rel_xyz(3), rel_rot6d(6)].
    returns: (T, 6) absolute pose6, with T_abs = T_base @ T_rel.
    """
    rel9 = np.asarray(rel9, dtype=np.float32)
    base6 = np.asarray(base6, dtype=np.float32)
    if base6.ndim == 1:
        base6 = np.broadcast_to(base6, rel9.shape[:-1] + (6,))
    T_rel = np.broadcast_to(np.eye(4, dtype=np.float32), rel9.shape[:-1] + (4, 4)).copy()
    T_rel[..., :3, :3] = rot6d_to_mat(rel9[..., 3:9])
    T_rel[..., :3, 3] = rel9[..., :3]
    T_abs = pose6_to_mat(base6) @ T_rel
    return mat_to_pose6(T_abs)


def abs_pose6_to_pose9(pose6: np.ndarray) -> np.ndarray:
    """Absolute state conversion: (..., 6) pose6 -> (..., 9) = [xyz(3), abs rot6d(6)].

    Per-frame (no base). Lifts the absolute arm rotation rotvec -> rot6d for the
    model's proprio/state input (T-Rex-style 9D absolute state). State is
    input-only, so no deploy inverse is needed.
    """
    pose6 = np.asarray(pose6, dtype=np.float32)
    R = pose6_to_mat(pose6)[..., :3, :3]
    return np.concatenate([pose6[..., :3], mat_to_rot6d(R)], axis=-1).astype(np.float32)


# --------------------------------------------------------------------------- #
# self-tests
# --------------------------------------------------------------------------- #
def _run_tests() -> None:
    _require_scipy()
    rng = np.random.default_rng(0)
    atol = 1e-4

    # (a) matrix round-trip is exact for valid SO(3)
    R = _Rotation.random(16, random_state=1).as_matrix().astype(np.float32)
    R2 = rot6d_to_mat(mat_to_rot6d(R))
    assert np.allclose(R2, R, atol=atol), "rot6d_to_mat(mat_to_rot6d(R)) != R"

    # (b) 6D is idempotent only AFTER canonicalization (NOT mat_to_rot6d(rot6d_to_mat(x))==x)
    x = rng.standard_normal((16, 6)).astype(np.float32)
    x_canon = mat_to_rot6d(rot6d_to_mat(x))
    assert np.allclose(rot6d_to_mat(x_canon), rot6d_to_mat(x), atol=atol), "rot6d not idempotent after canon"

    # (c) output is a proper rotation
    Rc = rot6d_to_mat(x)
    I = np.broadcast_to(np.eye(3, dtype=np.float32), Rc.shape)
    assert np.allclose(np.swapaxes(Rc, -1, -2) @ Rc, I, atol=atol), "R^T R != I"
    assert np.allclose(np.linalg.det(Rc), 1.0, atol=atol), "det(R) != 1"

    # pose6 <-> mat round-trip
    pose6 = np.concatenate([rng.standard_normal((16, 3)),
                            _Rotation.random(16, random_state=2).as_rotvec()], axis=-1).astype(np.float32)
    assert np.allclose(mat_to_pose6(pose6_to_mat(pose6))[:, :3], pose6[:, :3], atol=atol)
    R_from_pose = pose6_to_mat(pose6)[:, :3, :3]
    R_round = pose6_to_mat(mat_to_pose6(pose6_to_mat(pose6)))[:, :3, :3]
    assert np.allclose(R_round, R_from_pose, atol=atol), "pose6 rotation round-trip failed"

    # relativize -> compose recovers absolute target (compare on rotation MATRIX, not rotvec)
    base6 = np.concatenate([rng.standard_normal(3),
                            _Rotation.random(1, random_state=3).as_rotvec()[0]], axis=-1).astype(np.float32)
    targets6 = np.concatenate([rng.standard_normal((8, 3)),
                               _Rotation.random(8, random_state=4).as_rotvec()], axis=-1).astype(np.float32)
    rel9 = relativize_pose6_to_rot6d(base6, targets6)
    rec6 = compose_rot6d_to_pose6(base6, rel9)
    assert np.allclose(rec6[:, :3], targets6[:, :3], atol=atol), "relative compose xyz mismatch"
    R_tgt = pose6_to_mat(targets6)[:, :3, :3]
    R_rec = pose6_to_mat(rec6)[:, :3, :3]
    assert np.allclose(R_rec, R_tgt, atol=atol), "relative compose rotation mismatch"

    # abs_pose6_to_pose9 rotation matrix round-trip (rotvec compare avoided)
    pose9 = abs_pose6_to_pose9(pose6)
    R_state = rot6d_to_mat(pose9[:, 3:9])
    assert np.allclose(pose9[:, :3], pose6[:, :3], atol=atol)
    assert np.allclose(R_state, pose6_to_mat(pose6)[:, :3, :3], atol=atol), "state rot6d round-trip failed"

    print("pose_math: all self-tests passed")


if __name__ == "__main__":
    _run_tests()
