import pytorch_lightning as pl
from torch import nn
import torch
import torch.nn.functional as F
import torchmetrics.functional as metrics
from vit_pytorch.vit_for_small_dataset import ViT


class KilterModel(pl.LightningModule):
    def __init__(self, embedding_dim:int=256, dim:int=1024, depth:int=4, heads:int=8, mlp_dim:int=512,dropout:float=0.1, lr:float=1e-4, **kwargs):
        super().__init__()
        self.vit = ViT(
            image_size=48,
            channels=5,
            patch_size=8,
            num_classes=embedding_dim,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=dropout,
        )
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, 1))
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit(x)
        return self.mlp(x)

    def top1_top3_accuracy(self, preds, difficulty):
        diff = (preds - difficulty).abs()
        num_samples = diff.size(0)
        top1_guesses = (diff < 1).long()
        top3_guesses = (diff < 3).long()
        return top1_guesses.sum() / num_samples, top3_guesses.sum() / num_samples

    def shared_step(self, batch, step: str):
        x, difficulty = batch
        difficulty = difficulty.float()
        preds = self.forward(x).squeeze(-1)
        loss = F.mse_loss(preds, difficulty)
        mae = metrics.mean_absolute_error(preds, difficulty)
        spear = metrics.spearman_corrcoef(preds, difficulty)
        pears = metrics.pearson_corrcoef(preds, difficulty)
        r2 = metrics.r2_score(preds, difficulty)
        top1, top3 = self.top1_top3_accuracy(preds, difficulty)
        self.log(f"{step}/loss", loss)
        self.log(f"{step}/mae", mae)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
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
