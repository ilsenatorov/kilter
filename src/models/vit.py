import pytorch_lightning as pl
from torch import nn
import torch
import torch.nn.functional as F
import torchmetrics.functional as metrics
from vit_pytorch.vit_for_small_dataset import ViT


class KilterModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.vit = ViT(
            image_size=50,
            channels=4,
            patch_size=10,
            num_classes=config["embedding_dim"],
            dim=config["dim"],
            depth=config["depth"],
            heads=config["heads"],
            mlp_dim=config["mlp_dim"],
            dropout=0.1,
            emb_dropout=0.1,
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
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
