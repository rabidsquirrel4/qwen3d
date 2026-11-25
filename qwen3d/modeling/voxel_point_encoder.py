"""PointNet over the raw points inside each voxel token, fused into the voxel's image feature.

Qwen-3D's voxel tokens average ViT features that cover ~28 px each, so their geometry is far
coarser than the 5 cm voxels they sit in. ``VoxelPointEncoder`` adds what the points know:
every raw point (``VlmCdDatasetMapper.voxel_points``: the map cloud for the before scan, the
full-resolution depth for the after views) is matched to its voxel by integer cell key,
encoded from its position inside the cell and its colour, and max-pooled per voxel (as in
VoxelNet / PointPillars). The pooled vector and the voxel's point density go through a
zero-initialised projection to the LLM width and are added to the voxel's feature, so a new
encoder starts from exactly the pretrained model.

With ``VLMCD.MAP_SOURCE points`` the map has no images at all: ``map_tokens`` builds each
map voxel's whole feature from its points, through the same point trunk (so map and view
geometry are encoded alike) and its own head. That head ends in a parameter-free LayerNorm
times a gain set, on the first batch, to the RMS of the image voxel features, so map tokens
start at the scale the LLM is used to.
"""

import math

import torch
import torch.distributed as dist
from torch import nn
from torch_scatter import scatter_max, scatter_mean

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
    map_tokens : bool
        Also build the head for whole map tokens (``map_tokens``).
    """

    def __init__(self, out_dim, hidden_dim=128, voxel_size=0.05, max_points=32, map_tokens=False):
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
        self.map_head = None
        if map_tokens:
            # Max- and mean-pooled point features and the log point count -> a full token.
            self.map_head = nn.Sequential(
                nn.Linear(2 * hidden_dim + 1, 1024), nn.GELU(),
                nn.Linear(1024, out_dim),
                nn.LayerNorm(out_dim, elementwise_affine=False),
            )
            self.map_gain = nn.Parameter(torch.ones(()))
            self.register_buffer("map_gain_initialized", torch.zeros((), dtype=torch.bool))

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

    def map_tokens(self, points_xyz, points_rgb, points_voxel, voxel_count, voxel_size, reference):
        """Whole tokens for map voxels, from their points only.

        Parameters
        ----------
        points_xyz : torch.Tensor
            ``(P, 3)`` map points, metres.
        points_rgb : torch.Tensor
            uint8 ``(P, 3)``.
        points_voxel : torch.Tensor
            int64 ``(P,)`` voxel of each point (``0..M-1``).
        voxel_count : torch.Tensor
            ``(M,)`` points per voxel before subsampling.
        voxel_size : float
            Map voxel size (cells are ``round(xyz / voxel_size)``).
        reference : torch.Tensor
            Image voxel features ``(N, F)``; their RMS sets the token scale on the first call.

        Returns
        -------
        torch.Tensor
            float32 ``(M, F)``.
        """
        assert self.map_head is not None, "built without map_tokens=True"
        m = voxel_count.shape[0]
        with torch.autocast(device_type=points_xyz.device.type, enabled=False):
            p = points_xyz.float() / voxel_size
            rel = p - torch.round(p)
            rgb = points_rgb.float() / 255.0 - 0.5
            dtype = self.map_head[0].weight.dtype
            feats = self.point_mlp(torch.cat([rel, rgb], 1).to(dtype)).float()
            mx = scatter_max(feats, points_voxel, dim=0, dim_size=m)[0]
            mean = scatter_mean(feats, points_voxel, dim=0, dim_size=m)
            density = (torch.log1p(voxel_count.float()) / 10.0)[:, None]
            tok = self.map_head(torch.cat([mx, mean, density], 1).to(dtype)).float()
            if self.training and not bool(self.map_gain_initialized):
                with torch.no_grad():
                    rms = reference.float().pow(2).mean().sqrt()
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(rms)
                        rms /= dist.get_world_size()
                    self.map_gain.fill_(float(rms))
                    self.map_gain_initialized.fill_(True)
            return tok * self.map_gain.float()
