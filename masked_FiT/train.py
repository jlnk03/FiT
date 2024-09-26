import os
import torch
import argparse
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from models.fit import FiT_models
from preprocess.iterators import ImageNetLatentIterator
import lightning as L
from torch.utils.data import DataLoader
from diffusers import DDIMScheduler
import torch.nn.functional as F
from ema import EMA
import time

#################################################################################
#                                  PyTorch Lightning Module                     #
#################################################################################

torch.set_float32_matmul_precision('high')

forward = []
backward = []
loader = []

class FiTModule(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.model = torch.compile(FiT_models[args.model](), mode="max-autotune")
        self.model = FiT_models[args.model]()

        self.automatic_optimization = True
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

        self.save_hyperparameters()

    def forward(self, x, y, pos, mask, t):
        return self.model(x, t=t, y=y, pos=pos, mask=mask)

    def training_step(self, batch, batch_idx):
        latent, label, pos, mask = batch
        model_kwargs = {'y': label, 'pos': pos, 'mask': mask}

        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latent.shape[0],), device=self.device)

        noise = torch.randn(latent.shape, device=self.device)

        x_t = self.noise_scheduler.add_noise(latent, noise, t)

        model_output = self.model(x_t, t=t, **model_kwargs)

        assert model_output.shape == noise.shape

        masked_model_output = model_output[mask]
        masked_target = noise[mask]

        loss = F.mse_loss(masked_model_output, masked_target)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        latent, label, pos, mask = batch
        model_kwargs = {'y': label, 'pos': pos, 'mask': mask}

        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latent.shape[0],), device=self.device)

        noise = torch.randn(latent.shape, device=self.device)

        x_t = self.noise_scheduler.add_noise(latent, noise, t)

        model_output = self.model(x_t, t=t, **model_kwargs)

        assert model_output.shape == noise.shape

        masked_model_output = model_output[mask]
        masked_target = noise[mask]

        loss = F.mse_loss(masked_model_output, masked_target)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

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
            drop_last=True,
            collate_fn=dataset.collate
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
            collate_fn=dataset.collate
        )
        return loader

#################################################################################
#                                 Main Function                                 #
#################################################################################

def main(args):
    seed_everything(args.global_seed)
    
    model = FiTModule(args)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.results_dir, "checkpoints"),
        save_top_k=-1,  # Save all models
        # every_n_train_steps=args.ckpt_every
        every_n_epochs=1
    )

    ema_callback = EMA(decay=0.9999)
    
    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, ema_callback],
        precision='bf16-mixed',
        accumulate_grad_batches=2,
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
