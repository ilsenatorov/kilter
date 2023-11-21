import argparse

import pytorch_lightning as pl
import torch

import wandb
from src.data.datamodules import KilterDataModule
from src.models.autoencoder import SimpleAutoEncoder

torch.set_float32_matmul_precision("medium")

# Define the command-line arguments
parser = argparse.ArgumentParser(description="Your program description")
parser.add_argument("--embedding_dim", type=int, default=256, help="Size of embedding for image")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
args = parser.parse_args()


config = vars(args)

model = SimpleAutoEncoder(**config)

dm = KilterDataModule(args.batch_size, args.num_workers)

wandb.init(config=config, project="kilter_autoencoder")

trainer = pl.Trainer(
    accelerator="gpu",
    devices=-1,
    logger=pl.loggers.WandbLogger(log_model=True, save_dir="logs"),
    callbacks=[
        pl.callbacks.EarlyStopping(
            monitor="val/loss",
            patience=40,
            verbose=False,
            mode="min",
            min_delta=0.1,
        ),
        pl.callbacks.ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1),
    ],
    max_epochs=1000,
    precision="bf16-mixed",
)

trainer.fit(model=model, datamodule=dm)
trainer.test(model=model, datamodule=dm)
