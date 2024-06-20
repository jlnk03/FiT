import os
import torch
import argparse
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models.dit import DiT_models
from preprocess.iterators import ImageNetLatentIterator
import lightning as L
from torch.utils.data import DataLoader
from copy import deepcopy
from collections import OrderedDict
from diffusers import DDIMScheduler
import torch.nn.functional as F

#################################################################################
#                                  PyTorch Lightning Module                     #
#################################################################################

class FiTModule(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = DiT_models[args.model]()
        self.ema = deepcopy(self.model)
        self.diffusion = create_diffusion(timestep_respacing="")
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")
        self.automatic_optimization = True
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

        self.save_hyperparameters()

    def forward(self, x, y, pos, mask, h, w, t):
        return self.model(x, t=t, y=y, pos=pos, mask=mask, h=h, w=w)

    def training_step(self, batch, batch_idx):
        latent, label, pos, mask, h, w = batch

        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latent.shape[0],), device=self.device)

        noise = torch.randn(latent.shape, device=self.device)

        x_t = self.noise_scheduler.add_noise(latent, noise, t)

        model_output = self.model(x=x_t, t=t, y=label)

        loss = F.mse_loss(model_output[mask], noise[mask]).mean()

        # Manual optimization
        # opt = self.optimizers()
        # opt.zero_grad()
        # self.manual_backward(loss)
        # opt.step()

        # self.update_ema(self.ema, self.model)

        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0)
        return optimizer

    @torch.no_grad()
    def update_ema(self, ema_model, model, decay=0.9999):
        ema_params = OrderedDict(ema_model.named_parameters())
        model_params = OrderedDict(model.named_parameters())

        for name, param in model_params.items():
            name = name.replace("module.", "")
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

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

#################################################################################
#                                 Main Function                                 #
#################################################################################

def main(args):
    seed_everything(args.global_seed)
    
    model = FiTModule(args)
    
    # Initialize W&B logger
    wandb_logger = WandbLogger(name="train", project="FiT")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.results_dir, "checkpoints"),
        save_top_k=-1,  # Save all models
        every_n_train_steps=args.ckpt_every
    )
    
    trainer = Trainer(
        max_epochs=args.epochs,
        # accelerator='ddp',
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        precision=16 if torch.cuda.is_available() else 32,
        accumulate_grad_batches=8,
        log_every_n_steps=args.log_every
    )

    model = torch.compile(model, mode="reduce-overhead")
    
    trainer.fit(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--feature-val-path", type=str, default="features_val")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="FiT-XL/2")
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
