"""
Transport of the Stage 1 rotation state between camera frames.

In multi-view Stage 1 every view's network predicts the object rotation in *its own* camera
frame, but the ODE state carries a single rotation (view 0's frame). Feeding that state to every
view as-is makes views > 0 read a rotation that is wrong for their camera. These helpers
re-express the state in each view's frame (from the DA3 extrinsics) and bring the predicted
velocity back to the view-0 frame.

Convention (verified against regressed single-view rotations, within ~12 deg):
    Rm_i = Rm_0 @ Rrel_i^T,   Rrel_i = M @ R_w2c_i @ R_w2c_0^T @ M,   M = diag(-1, -1, 1)
where Rm is the rotation matrix the pose decoder builds from the 6D rotation
(columns b1, b2, b3 = Gram-Schmidt of the 6D vector, see inference_utils.pose_decoder).
"""
from typing import Sequence

import numpy as np
import torch

from sam3d_objects.pipeline.inference_utils import ROTATION_6D_MEAN, ROTATION_6D_STD

ROTATION_KEY = "6drotation_normalized"

_M = np.diag([-1.0, -1.0, 1.0])
_FD_EPS = 1e-3  # finite-difference step for pushing velocities through the (nonlinear) frame map


def relative_rotations(extrinsics: Sequence[np.ndarray]) -> np.ndarray:
    """Rrel_i for every view, relative to view 0. extrinsics: (N,3,4)/(N,4,4) world-to-camera."""
    R = [np.asarray(e, dtype=np.float64)[:3, :3] for e in extrinsics]
    return np.stack([_M @ R[i] @ R[0].T @ _M for i in range(len(R))])


def _normalized_6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    rot6d = x * ROTATION_6D_STD.to(x) + ROTATION_6D_MEAN.to(x)
    b1 = torch.nn.functional.normalize(rot6d[..., 0:3], dim=-1)
    a2 = rot6d[..., 3:6]
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _matrix_to_normalized_6d(Rm: torch.Tensor) -> torch.Tensor:
    rot6d = torch.cat([Rm[..., :, 0], Rm[..., :, 1]], dim=-1)
    return (rot6d - ROTATION_6D_MEAN.to(Rm)) / ROTATION_6D_STD.to(Rm)


def _map_state(x: torch.Tensor, Rrel: torch.Tensor, inverse: bool) -> torch.Tensor:
    """Re-express a normalized 6D rotation: view 0 -> view i (or back when inverse=True)."""
    x64 = x.double()
    Rm = _normalized_6d_to_matrix(x64)
    Rm = Rm @ Rrel if inverse else Rm @ Rrel.transpose(-1, -2)
    return _matrix_to_normalized_6d(Rm)


class RotationTransport:
    """Per-view rotation state/velocity transport between view 0 and view i."""

    def __init__(self, extrinsics: Sequence[np.ndarray]):
        self.Rrel = torch.from_numpy(relative_rotations(extrinsics))  # (V,3,3) float64

    def _R(self, view_idx: int, like: torch.Tensor) -> torch.Tensor:
        return self.Rrel[view_idx].to(like.device)

    def state_to_view(self, x_ref: torch.Tensor, view_idx: int) -> torch.Tensor:
        """Rotation state (view-0 frame) expressed in view `view_idx`'s camera frame."""
        if view_idx == 0:
            return x_ref
        return _map_state(x_ref, self._R(view_idx, x_ref), inverse=False).to(x_ref.dtype)

    def velocity_to_ref(self, x_ref: torch.Tensor, v_i: torch.Tensor, view_idx: int) -> torch.Tensor:
        """Velocity predicted in view i's frame, brought back to view 0's frame.

        Pushes x_i + eps * v_i back through the inverse frame map and differences against the
        round-trip of x_i, which is the first-order pullback of v_i at the current state.
        """
        if view_idx == 0:
            return v_i
        R = self._R(view_idx, x_ref)
        x_i = _map_state(x_ref, R, inverse=False)
        v64 = v_i.double()
        back = _map_state(x_i + _FD_EPS * v64, R, inverse=True)
        base = _map_state(x_i, R, inverse=True)
        return ((back - base) / _FD_EPS).to(v_i.dtype)
