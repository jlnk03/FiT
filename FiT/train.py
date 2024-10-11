import argparse
import os
from typing import Tuple, Union

import lightning as L
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from diffusers.models import AutoencoderKL
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import AdvancedProfiler
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from diffusion import create_diffusion
from ema import EMA
from models.fit import FiT_models
from preprocess.iterators import ImageNetLatentIterator
from preprocess.pos_embed import precompute_freqs_cis_2d

#################################################################################
#                                  PyTorch Lightning Module                     #
#################################################################################

torch.set_float32_matmul_precision('high')


class FiTModule(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = torch.compile(FiT_models[args.model](), mode="max-autotune")

        self.automatic_optimization = True
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

        self.save_hyperparameters()

    def forward(self, x, y, pos, mask, t):
        return self.model(x, t=t, y=y, pos=pos, mask=mask)

    def training_step(self, batch, batch_idx):
        latent, label, pos, mask, h, w = batch
        model_kwargs = {'y': label, 'pos': pos, 'mask': mask}

        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latent.shape[0],), device=self.device)

        noise = torch.randn(latent.shape, device=self.device)

        x_t = self.noise_scheduler.add_noise(latent, noise, t)

        model_output = self.model(x_t, t=t, **model_kwargs)

        # loss = (torch.sum(torch.pow(model_output * mask - noise * mask, 2), dim=1)
        #         / torch.clamp(torch.sum(mask, dim=1), min=1)).mean()

        assert model_output.shape == noise.shape

        # Apply the mask to the model output and the target
        masked_model_output = model_output[mask]
        masked_target = noise[mask]

        loss = F.mse_loss(masked_model_output, masked_target)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        latent, label, pos, mask, h, w = batch
        model_kwargs = {'y': label, 'pos': pos, 'mask': mask}

        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latent.shape[0],), device=self.device)

        noise = torch.randn(latent.shape, device=self.device)

        x_t = self.noise_scheduler.add_noise(latent, noise, t)

        model_output = self.model(x_t, t=t, **model_kwargs)

        # loss = (torch.sum(torch.pow(model_output * mask - noise * mask, 2), dim=1)
        #         / torch.clamp(torch.sum(mask, dim=1), min=1)).mean()

        assert model_output.shape == noise.shape

        # Apply the mask to the model output and the target
        masked_model_output = model_output[mask]
        masked_target = noise[mask]

        loss = F.mse_loss(masked_model_output, masked_target)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _patchify(self, x: torch.Tensor, p: int) -> torch.Tensor:
        # N, C, H, W -> N, T, D
        n, c, h, w = x.shape
        nh, nw = h // p, w // p
        x = x.view(n, c, nh, p, nw, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(n, nh * nw, p * p * c)
        return x

    def _unpatchify(self, x: torch.Tensor, nh: int, nw: int, p: int, c: int) -> torch.Tensor:
        # N, T, D -> N, C, H, W
        n, _, _ = x.shape
        x = x.view(n, nh, nw, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(n, c, nh * p, nw * p)
        return x

    def _pad_latent(self, x: torch.Tensor, p: int, max_size: int, max_length: int) -> torch.Tensor:
        # N, C, H, W -> N, C, max_size, max_size
        n, c, _, _ = x.shape
        nh, nw = max_size // p, max_size // p

        x_fill = self._patchify(x, p)
        if x_fill.shape[1] > max_length:
            return x
        x_padded = torch.zeros((n, max_length, p * p * c), dtype=x.dtype, device=x.device)
        x_padded[:, : x_fill.shape[1]] = x_fill
        x_padded = self._unpatchify(x_padded, nh, nw, p, c)
        return x_padded

    def _unpad_latent(self, x: Tensor, valid_t: int, h: int, w: int, p: int) -> Tensor:
        # N, C, max_size, max_size -> N, C, H, W
        _, c, _, _ = x.shape
        nh, nw = h // p, w // p
        x = self._patchify(x, p)
        x = x[:, :valid_t]
        x = self._unpatchify(x, nh, nw, p, c)
        return x

    def _create_pos_embed(
            self, h: int, w: int, p: int, max_length: int, embed_dim: int, method: str = "rotate"
    ) -> Tuple[torch.Tensor, int]:
        # 1, T, D
        nh, nw = h // p, w // p
        # if method == "rotate":
        pos_embed_fill = precompute_freqs_cis_2d(embed_dim, nh, nw, max_length=max_length)
        # else:
        #     pos_embed_fill = get_2d_sincos_pos_embed(embed_dim, nh, nw)

        pos_embed_fill = torch.tensor(pos_embed_fill, device=self.device, dtype=torch.float32)

        if pos_embed_fill.shape[0] > max_length:
            pos_embed = pos_embed_fill
        else:
            pos_embed = torch.zeros((max_length, embed_dim), dtype=torch.float32)
            pos_embed[: pos_embed_fill.shape[0]] = pos_embed_fill

        pos_embed = pos_embed[None, ...]
        pos_embed = torch.tensor(pos_embed)
        return pos_embed, pos_embed_fill.shape[0]

    def _create_mask(self, valid_t: int, max_length: int, n: int) -> torch.Tensor:
        # 1, T
        if valid_t > max_length:
            mask = torch.ones((valid_t,), dtype=torch.bool)
        else:
            mask = torch.zeros((max_length,), dtype=torch.bool)
            mask[:valid_t] = True
        mask = mask.unsqueeze(0).repeat(n, 1)
        return mask

    def predict_step(self, x: Tensor, t: Tensor, y: Tensor, pos: Tensor, mask: Tensor, cfg_scale: Union[float, Tensor]):

        # Load model:
        latent_size = args.image_size // 8
        latent_height, latent_width = args.image_height // 8, args.image_width // 8
        model = self.model.load_from_checkpoint(args.ckpt or "checkpoint/FiT-B-2.pt")
        diffusion = create_diffusion(str(args.num_sampling_steps))
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(self.device)

        # Labels to condition the model with (feel free to change):
        class_labels = [207, 396, 372, 396, 88, 979, 417, 279]

        # Create sampling noise:
        n = len(class_labels)
        z = torch.randn(n, 4, latent_height, latent_width, device=self.device)
        y = torch.tensor(class_labels, device=self.device)
        y_null = torch.ones_like(y) * 1000

        y = torch.cat([y, y_null], 0)
        z = torch.cat([z, z], 0)

        p = 2
        max_size = 32
        max_length = 256

        ### mindspore infer pipeline
        n, _, h, w = z.shape
        z = self._pad_latent(z, p, max_size, max_length)
        pos, valid_t = self._create_pos_embed(h, w, p, max_length, 64, "rotate")
        mask = self._create_mask(valid_t, max_length, n)

        model_kwargs = dict(y=y, pos=pos, mask=mask, cfg_scale=15)

        # TODO: infer pipeline
        latents = diffusion.ddim_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=self.device
        )

        samples, _ = latents.chunk(2, dim=0)  # Remove null class samples
        samples = self._unpad_latent(samples, valid_t, h, w, p)

        samples = vae.decode(samples / 0.18215).sample

        # Save and display images:
        save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0)
        return optimizer

    def train_dataloader(self):
        dataset = ImageNetLatentIterator({
            "latent_folder": self.args.feature_path,
            "sample_size": 256,
            "patch_size": 2,
            "vae_scale": 8,
            "C": 4,
            "embed_dim": 16,
            "embed_method": "rotate"
        })
        loader = DataLoader(
            dataset,
            batch_size=self.args.global_batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        return loader

    def val_dataloader(self):
        dataset = ImageNetLatentIterator({
            "latent_folder": self.args.feature_val_path,
            "sample_size": 256,
            "patch_size": 2,
            "vae_scale": 8,
            "C": 4,
            "embed_dim": 16,
            "embed_method": "rotate"
        })
        loader = DataLoader(
            dataset,
            batch_size=self.args.global_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return loader


#################################################################################
#                                 Main Function                                 #
#################################################################################

def main(args):
    seed_everything(args.global_seed)

    model = FiTModule(args)

    # Initialize W&B logger
    wandb_logger = WandbLogger(name="FiT_Training_100_epochs", project="FiT", resume="allow", id=args.wandb_run_id)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.results_dir, "checkpoints"),
        save_top_k=-1,  # Save all models
        # every_n_train_steps=args.ckpt_every
        every_n_epochs=1
    )

    ema_callback = EMA(decay=0.9999)

    profiler = AdvancedProfiler(dirpath=args.results_dir, filename="perf_logs")

    trainer = Trainer(
        max_epochs=args.epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, ema_callback],
        precision='bf16-mixed',
        accumulate_grad_batches=2,
        profiler=profiler,
        log_every_n_steps=args.log_every,
    )

    trainer.fit(model, ckpt_path=args.resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--feature-val-path", type=str, default="features_val")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(FiT_models.keys()), default="FiT-B/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    args = parser.parse_args()
    main(args)
