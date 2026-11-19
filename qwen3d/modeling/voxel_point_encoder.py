"""PointNet over the raw points inside each voxel token, fused into the voxel's image feature.

Qwen-3D's voxel tokens average ViT features that cover ~28 px each, so their geometry is far
coarser than the 5 cm voxels they sit in. ``VoxelPointEncoder`` adds what the points know:
every raw point (``VlmCdDatasetMapper.voxel_points``: the map cloud for the before scan, the
full-resolution depth for the after views) is matched to its voxel by integer cell key,
encoded from its position inside the cell and its colour, and max-pooled per voxel (as in
VoxelNet / PointPillars). The pooled vector and the voxel's point density go through a
zero-initialised projection to the LLM width and are added to the voxel's feature, so a new
encoder starts from exactly the pretrained model.
"""

import math

import torch
from torch import nn
from torch_scatter import scatter_max

_KEY_OFFSET, _KEY_RANGE = 1 << 20, 1 << 21


def cell_keys(xyz, voxel_size):
    """Integer key of the voxel cell holding each point (``round(xyz / voxel_size)``).

    Twin of ``dataset_mapper_vlm_cd.voxel_cell_keys``.

    Parameters
    ----------
    xyz : torch.Tensor
        ``(N, 3)``.
    voxel_size : float

    Returns
    -------
    tuple of torch.Tensor
        int64 keys ``(N,)`` and the integer cell indices ``(N, 3)``.
    """
    cell = torch.round(xyz.float() / voxel_size).long()
    c = cell + _KEY_OFFSET
    return (c[:, 0] * _KEY_RANGE + c[:, 1]) * _KEY_RANGE + c[:, 2], cell


class VoxelPointEncoder(nn.Module):
    """Per-voxel PointNet whose output is added to the voxel tokens.

    Parameters
    ----------
    out_dim : int
        Width of the voxel features (the LLM hidden size).
    hidden_dim : int
        Width of the per-point features.
    voxel_size : float
        Metres; must match the voxelization of the tokens.
    max_points : int
        Points per voxel the mapper keeps at most (normalises the density feature).
    """

    def __init__(self, out_dim, hidden_dim=128, voxel_size=0.05, max_points=32):
        super().__init__()
        self.voxel_size = float(voxel_size)
        self.max_points = int(max_points)
        # Point input: position inside its cell (in cell units, -0.5..0.5) and RGB (-0.5..0.5).
        self.point_mlp = nn.Sequential(
            nn.Linear(6, 64), nn.GELU(),
            nn.Linear(64, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim + 1)
        self.proj = nn.Linear(hidden_dim + 1, out_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, voxel_xyz, points_xyz, points_rgb):
        """Feature offsets for the voxel tokens.

        Parameters
        ----------
        voxel_xyz : torch.Tensor
            ``(N, 3)`` voxel centres, in the coordinates that were voxelized (the after scan
            still shifted), so ``round(xyz / voxel_size)`` is each voxel's cell.
        points_xyz : torch.Tensor
            ``(P, 3)`` raw points, same coordinates.
        points_rgb : torch.Tensor
            uint8 ``(P, 3)``.

        Returns
        -------
        tuple
            ``(offsets, stats)``: float32 ``(N, out_dim)`` (zero for voxels without
            points) and a dict with ``points_used`` and ``voxels_with_points``.
        """
        n = voxel_xyz.shape[0]
        vkeys, _ = cell_keys(voxel_xyz, self.voxel_size)
        pkeys, pcell = cell_keys(points_xyz, self.voxel_size)
        sorted_keys, perm = vkeys.sort()
        pos = torch.searchsorted(sorted_keys, pkeys).clamp(max=n - 1)
        hit = sorted_keys[pos] == pkeys
        vid = perm[pos[hit]]

        with torch.autocast(device_type=voxel_xyz.device.type, enabled=False):
            rel = points_xyz[hit].float() / self.voxel_size - pcell[hit].float()
            rgb = points_rgb[hit].float() / 255.0 - 0.5
            dtype = self.proj.weight.dtype
            feats = self.point_mlp(torch.cat([rel, rgb], 1).to(dtype)).float()
            pooled = scatter_max(feats, vid, dim=0, dim_size=n)[0]
            count = torch.bincount(vid, minlength=n).float()
            density = torch.log1p(count) / math.log1p(self.max_points)
            out = self.proj(self.norm(torch.cat([pooled, density[:, None]], 1).to(dtype))).float()
            out = out * (count > 0).float()[:, None]
        return out, {"points_used": int(hit.sum()), "voxels_with_points": int((count > 0).sum())}
