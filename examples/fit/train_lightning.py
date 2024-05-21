import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pytorch_lightning import LightningModule, Trainer, seed_everything, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
import argparse
import os

from models import FiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


class DiTModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = FiT_models[args.model](
            input_size=args.image_size // 8,
            num_classes=args.num_classes
        )
        self.ema = deepcopy(self.model)
        self.diffusion = create_diffusion(timestep_respacing="")
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0)

    def update_ema(self, decay=0.9999):
        ema_params = OrderedDict(self.ema.named_parameters())
        model_params = OrderedDict(self.model.named_parameters())
        for name, param in model_params.items():
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
        t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=self.device)
        model_kwargs = dict(y=y)
        loss_dict = self.diffusion.training_losses(self.model, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()
        self.update_ema()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_start(self):
        self.model.train()
        self.ema.eval()

    def on_train_epoch_end(self):
        if self.trainer.is_global_zero:
            checkpoint = {
                "model": self.model.state_dict(),
                "ema": self.ema.state_dict(),
                "opt": self.optimizers().state_dict(),
                "args": self.args
            }
            checkpoint_path = f"{self.trainer.logger.log_dir}/checkpoints/epoch-{self.current_epoch:03d}.pt"
            torch.save(checkpoint, checkpoint_path)

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        dataset = ImageFolder(self.args.data_path, transform=transform)
        return DataLoader(
            dataset,
            batch_size=self.args.global_batch_size // torch.cuda.device_count(),
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True
        )


def main(args):
    seed_everything(args.global_seed)

    logger = loggers.TensorBoardLogger(save_dir=args.results_dir, name="DiT")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{logger.log_dir}/checkpoints",
        save_top_k=-1,  # Save all checkpoints
        every_n_train_steps=args.ckpt_every
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        log_every_n_steps=args.log_every,
        precision=16
    )

    model = DiTModel(args)
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
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
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    args = parser.parse_args()
    main(args)