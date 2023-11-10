import argparse
import pytorch_lightning as pl
import torch
import wandb
from src.models.predict_vit import KilterModel
from src.data.datamodules import KilterDataModule

torch.set_float32_matmul_precision("medium")

# Define the command-line arguments
parser = argparse.ArgumentParser(description="Your program description")
parser.add_argument("--embedding_dim", type=int, default=256, help="Size of embedding for image")
parser.add_argument("--dim", type=int, default=1024, help="Model dimension")
parser.add_argument("--depth", type=int, default=4, help="Model depth")
parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
parser.add_argument("--mlp_dim", type=int, default=512, help="MLP dimension")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
args = parser.parse_args()


config = vars(args)

model = KilterModel(config)

dm = KilterDataModule(args.batch_size, args.num_workers)

wandb.init(config=config, project="kilter", tags=["small_datasets"])

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
