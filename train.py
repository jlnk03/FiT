import os
import datetime
import logging
import yaml
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from torchvision import datasets, transforms

from models import FiTModels
FiTModel = FiTModels["FiT-B/2"]

from diffusion import create_diffusion

class FiTTrainingModule(pl.LightningModule):
    def __init__(self):
        super(FiTTrainingModule, self).__init__()
        self.model = FiTModel(
        )
        self.diffusion = create_diffusion()

    def forward(self, x):
        #TODO: pass correct parameters such as mask, noise, etc.
        return self.model.construct(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = self.compute_loss(output, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.start_learning_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.hparams.decay_steps, gamma=0.1
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageNet(self.hparams.data_path, transform=transform)
        return DataLoader(dataset, batch_size=self.hparams.train_batch_size, shuffle=True, num_workers=self.hparams.num_parallel_workers)

    def compute_loss(self, outputs, targets):
        # Define the loss computation
        return torch.nn.functional.mse_loss(outputs, targets)

def main(args):
    # Initialize logger
    # logger = TensorBoardLogger(save_dir=args.output_path, name="fit_logs")

    # Initialize model
    model = FiTTrainingModule(args)

    # Initialize callbacks
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=args.output_path, save_top_k=args.ckpt_max_keep, monitor="val_loss", mode="min"
    # )
    # lr_monitor = LearningRateMonitor(logging_interval="step")

    # Initialize Trainer
    trainer = Trainer(
        max_epochs=1000,
        accelerator='auto',
        devices='auto',
        precision=16,
        # logger=logger,
        # callbacks=[checkpoint_callback, lr_monitor]
    )

    # Train the model
    trainer.fit(model)

if __name__ == "__main__":
    parser = ArgumentParser()
    # Add arguments from MindSpore script as needed
    args = parser.parse_args()
    main(args)
