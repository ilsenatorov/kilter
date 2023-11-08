import pytorch_lightning as pl
from vit_pytorch import SimpleViT
from torch import nn
import torch
import torch.nn.functional as F
from src.data.datasets import KilterDataset
from torch.utils.data import DataLoader


class KilterModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.vit = SimpleViT(
            image_size=50,
            patch_size=10,
            num_classes=256,
            dim=1024,
            depth=4,
            heads=8,
            mlp_dim=512,
            channels=4,
        )
        self.angle_mlp = nn.Sequential(nn.Linear(1, 16), nn.ReLU())
        self.combined_mlp = nn.Sequential(nn.Linear(256 + 16, 1))

    def forward(self, x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        x = self.vit(x)
        angle_x = self.angle_mlp(angle.unsqueeze(-1).float())
        x = torch.cat([x, angle_x], dim=1)
        return self.combined_mlp(x)

    def shared_step(self, batch, step: str):
        x, angle, difficulty = batch
        preds = self.forward(x.float(), angle.float())
        loss = F.mse_loss(preds, difficulty.float().unsqueeze(-1))
        self.log(f"{step}/loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


model = KilterModel()

train = KilterDataset(split="train")
val = KilterDataset(split="val")
test = KilterDataset(split="test")

BATCH_SIZE = 32

train_dl = DataLoader(train, batch_size=BATCH_SIZE)
val_dl = DataLoader(val, batch_size=BATCH_SIZE)
test_dl = DataLoader(test, batch_size=BATCH_SIZE)


trainer = pl.Trainer(
    accelerator="gpu",
    devices=-1,
    logger=pl.loggers.WandbLogger(log_model=True, project="kilter", save_dir="logs"),
    max_epochs=10,
)


trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)
