from typing import Optional

import torch

__all__ = [
    "get_1d_sincos_pos_embed",
    "get_2d_sincos_pos_embed",
    "precompute_freqs_cis_2d"
]


def get_2d_sincos_pos_embed(embed_dim: int, nh: int, nw: Optional[int] = None) -> torch.Tensor:
    """Generate 2D sinusoidal positional embedding based on the given height and width
    referred from https://github.com/facebookresearch/mae

    Args:
        embed_dim: embedding dimension.
        nh: image height
        nw: image width. If it is not given, then `nw` is equal to `nh`. Default: None
    """
    nw = nh if nw is None else nw

    grid_h = torch.arange(nh, dtype=torch.float)
    grid_w = torch.arange(nw, dtype=torch.float)

    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')  # here w goes first
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, nh, nw])

    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    return pos_embed


def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> torch.Tensor:
    """
    Generate sinusoidal/cosinusoidal positional embeddings for 1D data.

    Args:
        embed_dim (int): The dimensionality of the embeddings.
        length (int): The length of the 1D data.

    Returns:
        numpy.ndarray: The positional embeddings of shape (length, embed_dim).
    """
    pos = torch.arange(0, length).reshape((-1, 1))

    return _get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def precompute_freqs_cis_2d(
        dim: int, nh: int, nw: Optional[int] = None, theta: float = 10000.0, max_length: Optional[int] = None
) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponentials (cis) with given dimensions, for 2D RoPE
    referered from 1D RoPE https://github.com/meta-llama/llama and paper `FiT` https://arxiv.org/abs/2402.12376

    If max_length is not None, then a length extrapolation algo. `VisionNTK` from `FiT` will be used for tensor calculation.

    Args:
        dim: dimension of the frequency tensor
        nh: image height
        nw: image width. If it is not given, then `nw` is equal to `nh`. Default: None
        theta: Scaling factor for frequency computation. Defaults: 10000.0.
        max_length: If it is None, then the VisionNTK algo. will be applied. Default: None
    """
    nw = nh if nw is None else nw

    grid_h = torch.arange(nh, dtype=torch.float)
    grid_w = torch.arange(nw, dtype=torch.float)

    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')  # here w goes first
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, nh, nw])

    freqs_cis = _precompute_freqs_cis_2d_from_grid(dim, grid, theta=theta, max_length=max_length)  # (M, D/2, 2)
    freqs_cis = torch.reshape(freqs_cis, (freqs_cis.shape[0], -1))

    return freqs_cis


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.concatenate([emb_w, emb_h], dim=1)  # (H*W, D)

    return emb


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0

    omega = torch.arange(embed_dim // 2, dtype=torch.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    out = torch.outer(torch.flatten(pos), torch.flatten(omega))  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.concatenate([emb_sin, emb_cos], dim=1)  # (M, D)

    return emb


def _precompute_freqs_cis_2d_from_grid(
        dim: int, grid: torch.Tensor, theta: float = 10000.0, max_length: Optional[int] = None
) -> torch.Tensor:
    freqs_cis_w = _precompute_freqs_cis_1d_from_grid(dim // 2, grid[0], theta=theta, max_length=max_length)
    freqs_cis_h = _precompute_freqs_cis_1d_from_grid(dim // 2, grid[1], theta=theta, max_length=max_length)

    freqs_cis = torch.concatenate([freqs_cis_w, freqs_cis_h], dim=1)

    return freqs_cis


def _precompute_freqs_cis_1d_from_grid(
        dim: int, pos: torch.Tensor, theta: float = 10000.0, max_length: Optional[int] = None
) -> torch.Tensor:
    if max_length is not None:
        # VisionNTK
        s = max((torch.max(pos) / torch.sqrt(torch.Tensor([max_length]))).item(), 1.0)
        theta = theta * torch.power(s, dim / (dim - 2))

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float)[: (dim // 2)] / dim))
    freqs = torch.outer(torch.flatten(pos), torch.flatten(freqs))

    a = torch.cos(freqs)
    b = torch.sin(freqs)  # represent for a + ib

    freqs_cis = torch.stack([a, b], dim=-1)

    return freqs_cis
