import argparse

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import wandb
from src.data.datasets import KilterDiffusionDataset
from src.models.diffusion import DiffusionModel, GuidedDiffusionModel, SimpleDiffusionModel

# torch.set_float32_matmul_precision("medium")

parser = argparse.ArgumentParser(description="Your program description")
parser.add_argument("--model", type=str, default="normal", choices=["normal", "guided", "simple"])
parser.add_argument("--dim", type=int, default=64, help="Model dimension")
parser.add_argument("--timesteps", type=int, default=1000, help="Timesteps")
parser.add_argument("--objective", type=str, default="pred_v", help="Objective")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
args = parser.parse_args()


# Create config dictionary from command-line arguments
config = vars(args)

model_types = {"normal": DiffusionModel, "guided": GuidedDiffusionModel, "simple": SimpleDiffusionModel}
model = model_types[args.model](**config)

ds = KilterDiffusionDataset()
dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

wandb.init(config=config, project="kilter_diffusion")

trainer = pl.Trainer(
    accelerator="gpu",
    devices=-1,
    logger=pl.loggers.WandbLogger(log_model=True, save_dir="logs"),
    max_epochs=1000,
    # precision="",
    callbacks=[
        # pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2),
        pl.callbacks.ModelCheckpoint(monitor="train/loss", mode="min", save_top_k=1),
        pl.callbacks.EarlyStopping(monitor="train/loss", mode="min", patience=10),
    ],
)

trainer.fit(model=model, train_dataloaders=dl)
