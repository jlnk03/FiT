import argparse
import os
from collections import OrderedDict

import lightning as L
import torch
import torch._dynamo

from diffusers import DDIMScheduler
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from models.fit import FiT_models
from preprocess.iterators import ImageNetLatentIterator


#################################################################################
#                                  PyTorch Lightning Module                     #
#################################################################################

torch.set_float32_matmul_precision('high')

class FiTModule(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = torch.compile(FiT_models[args.model](), mode="max-autotune")
        # torch.compile with max-autotune only on the model not on the LightningModule

        # self.ema = deepcopy(self.model)
        # see https://github.com/Lightning-AI/pytorch-lightning/issues/8100#issuecomment-867819299
        # else the parameters are also taken into account for training

        # self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")
        # We could either just load the vae in the beginning of the validation or implement some offloading
        # (it can't be found by PyTorch Ligtning else it is moved automatically). It isn't needed during training

        self.automatic_optimization = True
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
        #
        # LOOKUP: Mindone uses betas, huge part missing here
        #
        # Did they use DDIM with 1000 steps in FiT? Did they use the default betas in FiT?

        self.save_hyperparameters()

    def forward(self, x, y, pos, mask, h, w, t):
        return self.model(x, t=t, y=y, pos=pos, mask=mask, h=h, w=w)

    def training_step(self, batch, batch_idx):
        latent, label, pos, mask, h, w = batch
        model_kwargs = {'y': label, 'pos': pos, 'mask': mask, 'h': h, 'w': w}

        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latent.shape[0],), device=self.device)
        #
        # LOOKUP: Mindone basically has the same line of code in training
        #
        # Do they use uniform timestep sampling?

        noise = torch.randn(latent.shape, device=self.device)

        x_t = self.noise_scheduler.add_noise(latent, noise, t)

        model_output = self.model(x_t, t=t, **model_kwargs)

        loss = (torch.sum(torch.pow(model_output * mask - noise * mask, 2), dim=1)
                / torch.clamp(torch.sum(mask, dim=1), min=1)).mean()
        # TODO: .sum() / mask.sum() do we need to take into account how much is masked?
        #
        # LOOKUP: Mindone uses MSE per image normalized by number of unmasked tokens,
        # masked tokens not included + something called vb and then calculates mean
        # The noise is not rescaled (Many options as ground truth, but noise is never rescaled)
        # A lot to do here
        #
        # is this the same loss that is used in FiT? Epsilon without rescaling?

        # opt = self.optimizers()
        # opt.zero_grad()
        # self.manual_backward(loss)
        # opt.step()

        # self.update_ema(self.ema, self.model)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def loss(self, input, noise, mask):
        diff = (torch.sum(input * mask, dim=1) / torch.sum(mask, dim=1) - torch.sum(noise * mask, dim=1) / torch.sum(mask, dim=1)).mean()

        return diff.mean()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0)
        # LOOKUP: Optimizer, learning_rate and weight decay correct corresponding to paper
        #
        # is this the same optimizer, learning rate scheduler, ... that is used in FiT?
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
    torch._dynamo.config.suppress_errors = True
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
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        precision="16-mixed" if torch.cuda.is_available() else 32,
        # If possible I would use bf16 but not possible on all GPUs
        accumulate_grad_batches=1,
        log_every_n_steps=args.log_every,

    )

    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="/storage/slurm/schnaus/latent_two/latent_two")
    # Put the dataset in the /storage/slurm/<username> folder for a faster access
    parser.add_argument("--feature-val-path", type=str, default="features_val")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(FiT_models.keys()), default="FiT-B/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    args = parser.parse_args()
    main(args)
