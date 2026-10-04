"""
Per-view alignment of the Stage 1 shape latent to the reference view's canonical frame.

Single-view SAM3D runs can put a near-symmetric object (a slab) in different canonical frames
depending on the view (e.g. long axis y vs z). Averaging velocities of such views fuses two
conventions into one crossed shape. The decoder is (approximately) equivariant to spatial
permutations of the 16^3 latent grid, so we find, per view, the cube rotation that maps that
view's own single-view occupancy onto the reference view's, and run the view's network in its
own frame: it sees A_i^-1(x_ref) and its shape velocity is mapped back with A_i before averaging.

Shape tokens are (B, 4096, C) with token index = flattened (x, y, z), z fastest (see
sample_sparse_structure_multi_view's decoder reshape).
"""
import itertools
from typing import List, Sequence, Tuple

import numpy as np
import torch
from loguru import logger

GRID = 16
# (perm, signs): new_axis_k = old_axis_perm[k], flipped where sign == -1
Spec = Tuple[Tuple[int, int, int], Tuple[int, int, int]]


def _det(perm, signs):
    sign_perm = np.linalg.det(np.eye(3)[list(perm)])
    return sign_perm * np.prod(signs)


def cube_rotations() -> List[Spec]:
    """The 24 proper rotations of the cube, identity first."""
    specs = [
        (perm, signs)
        for perm in itertools.permutations(range(3))
        for signs in itertools.product((1, -1), repeat=3)
        if _det(perm, signs) > 0
    ]
    specs.sort(key=lambda s: (s != ((0, 1, 2), (1, 1, 1)),))
    return specs


def apply_spec(a, spec: Spec, spatial_dims: Sequence[int]):
    """Apply a cube rotation to the three spatial dims of a tensor/array."""
    perm, signs = spec
    dims = list(spatial_dims)
    order = list(range(a.ndim))
    for k in range(3):
        order[dims[k]] = dims[perm[k]]
    a = a.permute(*order) if torch.is_tensor(a) else a.transpose(order)
    flip_dims = [dims[k] for k in range(3) if signs[k] < 0]
    if flip_dims:
        a = torch.flip(a, flip_dims) if torch.is_tensor(a) else np.flip(a, flip_dims)
    return a


def _apply_inverse(a, spec: Spec, spatial_dims: Sequence[int]):
    """Exact inverse of apply_spec: undo the flips first, then the permutation."""
    perm, signs = spec
    dims = list(spatial_dims)
    flip_dims = [dims[k] for k in range(3) if signs[k] < 0]
    if flip_dims:
        a = torch.flip(a, flip_dims) if torch.is_tensor(a) else np.flip(a, flip_dims)
    inv_perm = [0, 0, 0]
    for k in range(3):
        inv_perm[perm[k]] = k
    order = list(range(a.ndim))
    for k in range(3):
        order[dims[k]] = dims[inv_perm[k]]
    return a.permute(*order) if torch.is_tensor(a) else a.transpose(order)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


class LatentAligner:
    """specs[i] maps view i's canonical frame onto view 0's: A_i(occ_i) ~ occ_0."""

    def __init__(self, specs: List[Spec]):
        self.specs = specs

    @property
    def is_identity(self) -> bool:
        return all(s == self.specs[0] for s in self.specs) and self.specs[0] == ((0, 1, 2), (1, 1, 1))

    @classmethod
    def estimate(cls, occupancies: Sequence[np.ndarray], min_gain: float = 0.05) -> "LatentAligner":
        """occupancies: one boolean (64,64,64) grid per view, each from that view's own single-view run."""
        cands = cube_rotations()
        ref = occupancies[0]
        specs: List[Spec] = [cands[0]]
        for i in range(1, len(occupancies)):
            scores = [_iou(apply_spec(occupancies[i], c, (0, 1, 2)), ref) for c in cands]
            best = int(np.argmax(scores))
            chosen = cands[best] if scores[best] > scores[0] + min_gain else cands[0]
            logger.info(
                f"[LatentAlign] view {i}: IoU(identity)={scores[0]:.2f}, best={scores[best]:.2f} "
                f"{cands[best]} -> using {'identity' if chosen == cands[0] else chosen}"
            )
            specs.append(chosen)
        return cls(specs)

    def to_view(self, shape: torch.Tensor, view_idx: int) -> torch.Tensor:
        """Reference-frame shape tokens (B, N, C) -> view `view_idx`'s own frame."""
        spec = self.specs[view_idx]
        if view_idx == 0 or spec == self.specs[0]:
            return shape
        B, N, C = shape.shape
        g = shape.reshape(B, GRID, GRID, GRID, C)
        return _apply_inverse(g, spec, (1, 2, 3)).reshape(B, N, C)

    def from_view(self, shape: torch.Tensor, view_idx: int) -> torch.Tensor:
        """Shape tokens / velocity in view `view_idx`'s own frame -> reference frame."""
        spec = self.specs[view_idx]
        if view_idx == 0 or spec == self.specs[0]:
            return shape
        B, N, C = shape.shape
        g = shape.reshape(B, GRID, GRID, GRID, C)
        return apply_spec(g, spec, (1, 2, 3)).reshape(B, N, C)
