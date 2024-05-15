import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UniPCMultistepScheduler
import numpy as np
from typing import Any, Dict, Optional, Tuple
from torch import Tensor

from fit import FiT_models, apply_rotary_emb

from PIL import Image
from fit import FiT_models
import torchvision.transforms as transforms
from datasets import load_dataset


def _precompute_freqs_cis_1d_from_grid(
    dim: int, pos: np.ndarray, theta: float = 10000.0, max_length: Optional[int] = None
) -> np.ndarray:
    if max_length is not None:
        # VisionNTK
        s = max(np.max(pos) / np.sqrt(max_length), 1.0)
        theta = theta * np.power(s, dim / (dim - 2))

    freqs = 1.0 / \
        (theta ** (np.arange(0, dim, 2, dtype=np.float32)[: (dim // 2)] / dim))
    freqs = np.outer(pos, freqs)
    a = np.cos(freqs)
    b = np.sin(freqs)  # represent for a + ib
    freqs_cis = np.stack([a, b], axis=-1)
    return freqs_cis


def _precompute_freqs_cis_2d_from_grid(
    dim: int, grid: np.ndarray, theta: float = 10000.0, max_length: Optional[int] = None
) -> np.ndarray:
    freqs_cis_w = _precompute_freqs_cis_1d_from_grid(
        dim // 2, grid[0], theta=theta, max_length=max_length)
    freqs_cis_h = _precompute_freqs_cis_1d_from_grid(
        dim // 2, grid[1], theta=theta, max_length=max_length)
    freqs_cis = np.concatenate([freqs_cis_w, freqs_cis_h], axis=1)
    return freqs_cis


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
    freqs_cis = _precompute_freqs_cis_2d_from_grid(
        dim, grid, theta=theta, max_length=max_length)  # (M, D/2, 2)
    freqs_cis = np.reshape(freqs_cis, (freqs_cis.shape[0], -1))
    return freqs_cis


def _create_pos_embed(
    self, h: int, w: int, p: int, max_length: int, embed_dim: int, method: str = "rotate"
) -> Tuple[Tensor, int]:
    # 1, T, D
    nh, nw = h // p, w // p
    if method == "rotate":
        pos_embed_fill = precompute_freqs_cis_2d(
            embed_dim, nh, nw, max_length=max_length)
    # else:
    #     pos_embed_fill = get_2d_sincos_pos_embed(embed_dim, nh, nw)

    if pos_embed_fill.shape[0] > max_length:
        pos_embed = pos_embed_fill
    else:
        pos_embed = np.zeros((max_length, embed_dim), dtype=np.float32)
        pos_embed[: pos_embed_fill.shape[0]] = pos_embed_fill

    pos_embed = pos_embed[None, ...]
    pos_embed = Tensor(pos_embed)
    return pos_embed, pos_embed_fill.shape[0]


def _create_mask(self, valid_t: int, max_length: int, n: int) -> Tensor:
    # 1, T
    if valid_t > max_length:
        mask = np.ones((valid_t,), dtype=np.bool_)
    else:
        mask = np.zeros((max_length,), dtype=np.bool_)
        mask[:valid_t] = True
    mask = np.tile(mask[None, ...], (n, 1))
    mask = Tensor(mask)
    return mask


class FiTFusion(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="text_encoder", use_safetensors=True
        )
        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True)
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        self.FiT = FiT_models['FiT-B/2']()
        self.generator = torch.Generator(device=self.device).manual_seed(42)

        self.dataset = load_dataset("imagenet-1k")
        self.transform = transforms.Compose([transforms.Lambda(lambda img: img.resize((256, img.height)) if img.width > 256 else img),
                                             transforms.Lambda(lambda img: img.resize((img.width, 256)) if img.height > 256 else img),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], inplace=True)])
        self.loss = torch.nn.MSELoss()
        self.inference_steps = 25

    def forward(self, prompts, inference_steps=25, guidance_scale=7.5, height=256, width=256):
        batch_size = len(prompts)

        text_embeddings = self.encode_text(prompts)

        latents = torch.randn(
            (batch_size, 4, 256, 256),
            generator=self.generator,
            device=self.device,
        )

        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(inference_steps)

        for t in tqdm(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            time_step = torch.tensor([t] * batch_size, device=self.device)

            rope = _create_pos_embed(height // 8, width // 8, 2, 256 ** 2 / 4, 256)

            mask = _create_mask(height * width / 4, 256 ** 2 / 4, batch_size)

            # TODO: pass correct parameters such as mask, noise, etc.
            noise_pred = self.unpatchify(self.FiT.construct(self.FiT.patchify(latent_model_input), time_step, rope, mask))

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents

        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        image = Image.fromarray(image)

        return image

    def training_step(self, batch):
        loss = self.forward_with_loss(batch)

        return loss

    def validation_step(self, batch):
        loss = self.forward_with_loss(batch)

        return loss

    def forward_with_loss(self, batch, guidance_scale=7.5):
        sample_images, prompts = batch

        # NOT USED ATM
        prompts_encoded = self.encode_text(prompts)

        sample_images_encoded = self.vae.encode(sample_images)

        B, C, H, W = sample_images_encoded.shape

        timesteps_np = np.random.choice(self.inference_steps, size=(B,))
        timesteps = torch.from_numpy(timesteps_np).long().to(self.device)

        noise = torch.randn(sample_images_encoded.shape) * self.scheduler.init_noise_sigma

        noisy_images = self.noise_scheduler.add_noise(sample_images_encoded, noise, timesteps)

        rope = _create_pos_embed(H // 8, W // 8, 2, 256 ** 2 / 4, 256)

        mask = _create_mask(H * W / 4, 256**2/4, B)

        noise_pred = self.unpatchify(self.FiT.construct(self.FiT.patchify(noisy_images), timesteps, rope, mask))

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        sample_images_predicted = self.scheduler.step(noise_pred, timesteps, noisy_images).prev_sample

        return self.loss(sample_images_encoded, sample_images_predicted)

    def encode_text(self, prompts):
        text_input = self.tokenizer(
            prompts, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input)

        uncond_input = self.tokenizer(
            "",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input)

        return torch.cat([uncond_embeddings.repeat(len(prompts), 1), text_embeddings], dim=1)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

    def train_dataloader(self):
        dataset = self.dataset['train']
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, transforms=self.transform)
        return dataloader

    def val_dataloader(self):
        dataset = self.dataset['validation']
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, transforms=self.transform)
        return dataloader

    def test_dataloader(self):
        dataset = self.dataset['test']
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, transforms=self.transform)
        return dataloader
