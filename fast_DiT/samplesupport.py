import os
import random
from typing import Dict, List, Tuple
from typing import Optional
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
ALLOWED_FORMAT = {".jpeg", ".jpg", ".bmp", ".png"}

def get_2d_sincos_pos_embed(embed_dim: int, nh: int, nw: Optional[int] = None) -> np.ndarray:
    """Generate 2D sinusoidal positional embedding based on the given height and width
    referred from https://github.com/facebookresearch/mae

    Args:
        embed_dim: embedding dimension.
        nh: image height
        nw: image width. If it is not given, then `nw` is equal to `nh`. Default: None
    """
    nw = nh if nw is None else nw
    grid_h = np.arange(nh, dtype=np.float32)
    grid_w = np.arange(nw, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, nh, nw])
    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> np.ndarray:
    """
    Generate sinusoidal/cosinusoidal positional embeddings for 1D data.

    Args:
        embed_dim (int): The dimensionality of the embeddings.
        length (int): The length of the 1D data.

    Returns:
        numpy.ndarray: The positional embeddings of shape (length, embed_dim).
    """
    pos = np.arange(0, length).reshape((-1, 1))
    return _get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def precompute_freqs_cis_2d(
        dim: int, nh: int, nw: Optional[int] = None, theta: float = 10000.0, max_length: Optional[int] = None
) -> np.ndarray:
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
    grid_h = np.arange(nh, dtype=np.float32)
    grid_w = np.arange(nw, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, nh, nw])
    freqs_cis = _precompute_freqs_cis_2d_from_grid(dim, grid, theta=theta, max_length=max_length)  # (M, D/2, 2)
    freqs_cis = np.reshape(freqs_cis, (freqs_cis.shape[0], -1))
    return freqs_cis


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_w, emb_h], axis=1)  # (H*W, D)
    return emb


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)
    out = np.outer(pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def _precompute_freqs_cis_2d_from_grid(
        dim: int, grid: np.ndarray, theta: float = 10000.0, max_length: Optional[int] = None
) -> np.ndarray:
    freqs_cis_w = _precompute_freqs_cis_1d_from_grid(dim // 2, grid[0], theta=theta, max_length=max_length)
    freqs_cis_h = _precompute_freqs_cis_1d_from_grid(dim // 2, grid[1], theta=theta, max_length=max_length)
    freqs_cis = np.concatenate([freqs_cis_w, freqs_cis_h], axis=1)
    return freqs_cis


def _precompute_freqs_cis_1d_from_grid(
        dim: int, pos: np.ndarray, theta: float = 10000.0, max_length: Optional[int] = None
) -> np.ndarray:
    if max_length is not None:
        # VisionNTK
        s = max(np.max(pos) / np.sqrt(max_length), 1.0)
        theta = theta * np.power(s, dim / (dim - 2))

    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32)[: (dim // 2)] / dim))
    freqs = np.outer(pos, freqs)
    a = np.cos(freqs)
    b = np.sin(freqs)  # represent for a + ib
    freqs_cis = np.stack([a, b], axis=-1)
    return freqs_cis


def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    sinusoid_inp = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq)
    sinusoid_inp = torch.cat(
        (torch.Tensor(sinusoid_inp, dtype=torch.float32), torch.Tensor(sinusoid_inp, dtype=torch.float32)), dim=-1)
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)


def rotate_every_two(x: torch.Tensor):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(start_dim=-2, end_dim=-1)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    sin = sin.unsqueeze(0).unsqueeze(1)
    cos = cos.unsqueeze(0).unsqueeze(1)
    return (tensor * cos) + (rotate_every_two(tensor) * sin)


def apply_2d_rotary_pos(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    sincos_h, sincos_w = freqs_cis
    sin_h, cos_h = torch.split(sincos_h, sincos_h.shape[-1] // 2, dim=-1)
    sin_w, cos_w = torch.split(sincos_w, sincos_w.shape[-1] // 2, dim=-1)
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    q1 = apply_rotary_pos_emb(q1, sin_h, cos_h)
    k1 = apply_rotary_pos_emb(k1, sin_h, cos_h)
    q2 = apply_rotary_pos_emb(q2, sin_w, cos_w)
    k2 = apply_rotary_pos_emb(k2, sin_w, cos_w)
    q = torch.cat([q1, q2], dim=-1)
    k = torch.cat([k1, k2], dim=-1)
    return q, k

######
def resize_call( img: Image.Image , max_size: int = 256, scale: int = 8   ) -> Image.Image:
    w, h = img.size
    image_area = w * h
    max_area = max_size * max_size
    if image_area > max_area:
        ratio = max_area / image_area
        new_w = w * np.sqrt(ratio)
        new_h = h * np.sqrt(ratio)
    else:
        new_w = w
        new_h = h

    round_w, round_h = (np.round(np.array([new_w, new_h]) / scale) * scale).astype(int).tolist()
    if round_w * round_h > max_area:
        round_w, round_h = (np.floor(np.array([new_w, new_h]) / scale) * scale).astype(int).tolist()

    round_w, round_h = max(round_w, scale), max(round_h, scale)
    img = img.resize((round_w, round_h), resample=Image.BICUBIC)
    return img


def inspect_images( root: str) -> List[str]:
    images_info = list()

    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            _, ext = os.path.splitext(f)
            if ext.lower() in ALLOWED_FORMAT:
                fpath = os.path.join(dirpath, f)
                images_info.append(fpath)

    if len(images_info) == 0:
        raise RuntimeError(f"Cannot find any image under `{root}`")

    images_info = sorted(images_info)
    return images_info

def len(image_paths):
    return len(image_paths)

def __getitem__(image_paths,index):
    path = image_paths[index]

    with Image.open(path) as f:
        img = f.convert("RGB")
    resize =  resize_call(max_size= 256, patch_size=2)
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    img = resize(img)
    img = transform(img)

    return img, path

def inspect_latent(root: str) -> List[Dict[str, str]]:
    latent_info = list()

    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            _, ext = os.path.splitext(f)
            if ext.lower() in ".npy":
                fpath = os.path.join(dirpath, f)
                latent_info.append(dict(path=fpath, label=os.path.basename(dirpath)))

    if len(latent_info) == 0:
        raise RuntimeError(f"Cannot find any image under `{root}`")

    latent_info = sorted(latent_info, key=lambda x: x["path"])
    return latent_info

def create_label_mapping( latent_info: List[Dict[str, str]]):
    labels = set([x["label"] for x in latent_info])
    labels = sorted(list(labels))
    labels = dict(zip(labels, np.arange(len(labels), dtype=np.int32)))
    return labels

#def length(self):
 #   return len(self.latent_info)

def random_horiztotal_flip( latent: np.ndarray,) -> np.ndarray:
    if random.random() < 0.5:
        # perform a random horizontal flip in latent domain
        # mimic the effect of horizontal flip in image (not exactly identical)
        latent = latent[..., ::-1]
    return latent

def patchify( latent: np.ndarray, patch_size: int = 2, embed_dim: int = 64, embed_method: str = "rotate") -> Tuple[np.ndarray, np.ndarray]:
    c, h, w = latent.shape
    nh, nw = h // patch_size, w // patch_size

    latent = np.reshape(latent, (c, nh, patch_size, nw, patch_size))
    latent = np.transpose(latent, (1, 3, 2, 4, 0))  # nh, nw, patch, patch, c
    latent = np.reshape(latent, (nh * nw, -1))  # nh * nw, patch * patch * c

    if embed_method == "rotate":
        pos = precompute_freqs_cis_2d(embed_dim, nh, nw).astype(np.float32)
    else:
        pos = get_2d_sincos_pos_embed(embed_dim, nh, nw).astype(np.float32)
    return latent, pos

def getitem(latent, number_of_tokens,patch_size,embed_dim,max_length):
    #x = latent_info[idx]
    ## latent  = noise
    #latent = np.load(x["path"])
    C = 4
    height, width = latent.shape[1:]
    
    #latent = random_horiztotal_flip(latent)
    latent, pos = patchify(latent)
    #label = create_label_mapping[x["label"]]
    mask = np.ones(latent.shape[0], dtype=np.bool_)

    latent = torch.tensor(latent)
    latent = latent[torch.randperm(latent.shape[0])]
    latent = latent[:number_of_tokens]
    latent = torch.nn.functional.pad(latent, (
        0, patch_size * patch_size * C - latent.shape[1], 0, number_of_tokens - latent.shape[0]))

    label = torch.tensor(label)

    pos = torch.tensor(pos)
    pos = torch.nn.functional.pad(pos, (
        0, embed_dim - pos.shape[1], 0, max_length - pos.shape[0]))

    mask = torch.tensor(mask)
    mask = torch.nn.functional.pad(mask, (0, max_length - mask.shape[0]))

    return latent, label, pos, mask, height, width


def create_dataloader_imagenet_preprocessing(dataset):
    #dataset = ImageNetWithPathIterator(config)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)

    return dataloader


def create_dataloader_imagenet_latent(dataset):
    #dataset = ImageNetLatentIterator(config)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256)

    return dataloader
