import os
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
import argparse
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models.fit import FiT_models
from preprocess.iterators import ImageNetLatentIterator
import lightning as L
from torch.utils.data import DataLoader
from copy import deepcopy
from collections import OrderedDict
from diffusers import DDIMScheduler
import torch.functional as F
from lightning.pytorch.profilers import AdvancedProfiler
from torch import Tensor
from typing import Any, Dict, Tuple, Type, Union
from torchvision.utils import save_image
from ema import EMA

#################################################################################
#                                  PyTorch Lightning Module                     #
#################################################################################

torch.set_float32_matmul_precision('high')

class FiTModule(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = FiT_models[args.model]()
        self.diffusion = create_diffusion(timestep_respacing="")
        self.automatic_optimization = True

        self.model = torch.compile(FiT_models[args.model](), mode="max-autotune")

        self.save_hyperparameters()

    def forward(self, x, y, pos, mask, h, w, t):
        return self.model(x, t=t, y=y, pos=pos, mask=mask, h=h, w=w)

    def training_step(self, batch, batch_idx):
        latent, label, pos, mask, h, w = batch
        t = torch.randint(0, self.diffusion.num_timesteps, (latent.shape[0],), device=self.device)
        model_kwargs = {'y': label, 'pos': pos, 'mask': mask, 'h': h, 'w': w}
        loss_dict = self.diffusion.training_losses(self.model, latent, t, model_kwargs)
        loss = loss_dict["loss"].mean()

        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        latent, label, pos, mask, h, w = batch
        t = torch.randint(0, self.diffusion.num_timesteps, (latent.shape[0],), device=self.device)
        model_kwargs = {'y': label, 'pos': pos, 'mask': mask, 'h': h, 'w': w}
        loss_dict = self.diffusion.training_losses(self.model, latent, t, model_kwargs)
        val_loss = loss_dict["loss"].mean()
        
        # Logging
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss
    
    def predict_step(self, x: Tensor, t: Tensor, y: Tensor, pos: Tensor, mask: Tensor, cfg_scale: Union[float, Tensor]):
            # Setup PyTorch:
            torch.manual_seed(args.seed)
            torch.set_grad_enabled(False)
            device = "cuda" if torch.cuda.is_available() else "cpu"

            if args.ckpt is None:
                assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
                assert args.image_size in [256, 512]
                assert args.num_classes == 1000

            # Load model:
            latent_size = args.image_size // 8
            FiT_model = FiT_models[args.model](
                input_size=latent_size,
                num_classes=args.num_classes
            ).to(device)
            # load a custom FiT checkpoint from train.py:
            # ckpt_path = args.ckpt or f"FiT-B-2.pt"
            # state_dict = find_model(ckpt_path)
            # model.load_state_dict(state_dict)
            model = FiT_model.load_from_checkpoint(args.ckpt or "checkpoint/FiT-B-2.pt")
            model.eval()  # important!
            diffusion = create_diffusion(str(args.num_sampling_steps))
            vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

            # Labels to condition the model with (feel free to change):
            class_labels = [207]

            # Create sampling noise:
            n = len(class_labels)
            z = torch.randn(n, 4, latent_size, latent_size, device=device)
            y = torch.tensor(class_labels, device=device)

            # Setup classifier-free guidance:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            # TODO: add mask and pos
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

            # Sample images:
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            samples = vae.decode(samples / 0.18215).sample

            # Save and display images:
            save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0)
        return optimizer

    # @torch.no_grad()
    # def update_ema(self, ema_model, model, decay=0.9999):
    #     ema_params = OrderedDict(ema_model.named_parameters())
    #     model_params = OrderedDict(model.named_parameters())

    #     for name, param in model_params.items():
    #         name = name.replace("module.", "")
    #         ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

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
            drop_last=True
        )
        return loader

#################################################################################
#                                 Main Function                                 #
#################################################################################

def main(args):
    seed_everything(args.global_seed)
    
    model = FiTModule(args)
    
    # Initialize W&B logger
    wandb_logger = WandbLogger(name="FiT_Training_new", project="FiT")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.results_dir, "checkpoints"),
        save_top_k=-1,  # Save all models
        every_n_train_steps=args.ckpt_every
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
        log_every_n_steps=args.log_every
    )
    
    trainer.fit(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--feature-val-path", type=str, default="features_val")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(FiT_models.keys()), default="FiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    args = parser.parse_args()
    main(args)
