import argparse
import pytorch_lightning as pl
from vit_pytorch import SimpleViT
from torch import nn
import torch
import torch.nn.functional as F
from src.data.datasets import KilterDataset
from torch.utils.data import DataLoader
import wandb
import torchmetrics.functional as metrics

torch.set_float32_matmul_precision("medium")

# Define the command-line arguments
parser = argparse.ArgumentParser(description="Your program description")
parser.add_argument(
    "--embedding_dim", type=int, default=256, help="Size of embedding for image"
)
parser.add_argument("--dim", type=int, default=1024, help="Model dimension")
parser.add_argument("--depth", type=int, default=4, help="Model depth")
parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
parser.add_argument("--mlp_dim", type=int, default=512, help="MLP dimension")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
args = parser.parse_args()


class KilterModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.vit = SimpleViT(
            image_size=50,
            channels=4,
            patch_size=5,
            num_classes=config["embedding_dim"],
            dim=config["dim"],
            depth=config["depth"],
            heads=config["heads"],
            mlp_dim=config["mlp_dim"],
        )
        self.angle_mlp = nn.Sequential(nn.Linear(1, 16), nn.ReLU())
        self.combined_mlp = nn.Sequential(nn.Linear(config["embedding_dim"] + 16, 1))

    def forward(self, x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        x = self.vit(x)
        angle_x = self.angle_mlp(angle.unsqueeze(-1).float())
        x = torch.cat([x, angle_x], dim=1)
        return self.combined_mlp(x)

    def top1_top3_accuracy(self, preds, difficulty):
        diff = (preds - difficulty).abs()
        num_samples = diff.size(0)
        top1_guesses = (diff < 1).long()
        top3_guesses = (diff < 3).long()
        return top1_guesses.sum() / num_samples, top3_guesses.sum() / num_samples

    def shared_step(self, batch, step: str):
        x, angle, difficulty = batch
        difficulty = difficulty.float()
        preds = self.forward(x.float(), angle.float()).squeeze(-1)
        loss = F.mse_loss(preds, difficulty)
        spear = metrics.spearman_corrcoef(preds, difficulty)
        pears = metrics.pearson_corrcoef(preds, difficulty)
        r2 = metrics.r2_score(preds, difficulty)
        top1, top3 = self.top1_top3_accuracy(preds, difficulty)
        self.log(f"{step}/loss", loss)
        self.log(f"{step}/spear", spear)
        self.log(f"{step}/pears", pears)
        self.log(f"{step}/r2", r2)
        self.log(f"{step}/top1", top1)
        self.log(f"{step}/top3", top3)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=args.lr)
        lr_scheduler = {
            "monitor": "val/loss",
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                verbose=True,
                factor=0.1,
                patience=20,
                mode="min",
            ),
        }
        return [optimizer], [lr_scheduler]


# Create config dictionary from command-line arguments
config = vars(args)

model = KilterModel(config)

train = KilterDataset(split="train")
val = KilterDataset(split="val")
test = KilterDataset(split="test")

train_dl = DataLoader(train, batch_size=args.batch_size, num_workers=args.num_workers)
val_dl = DataLoader(val, batch_size=args.batch_size, num_workers=args.num_workers)
test_dl = DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers)

wandb.init(config=config, project="kilter")

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
        )
    ],
    max_epochs=1000,
    precision="bf16-mixed",
)

trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)
trainer.test(model=model, dataloaders=test_dl)
