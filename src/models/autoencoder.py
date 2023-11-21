import pytorch_lightning as pl
import torch
from torch import nn


class SimpleAutoEncoder(pl.LightningModule):
    def __init__(self, embedding_dim: int = 256, lr: float = 1e-4, **kwargs):
        super().__init__()
        self.lr = lr
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # output: 16 x 24 x 24
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # output: 8 x 12 x 12
            nn.Linear(32 * 12 * 12, embedding_dim),
        )

        # Bottleneck layer

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 32 * 12 * 12),
            nn.ReLU(True),
            # Reshape layer to match the output of the encoder
            nn.Unflatten(1, (32, 12, 12)),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 4, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = x[:, :-1, :, :]  # ignore the angle
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return x

    def shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, *, step: str = "train"):
        x, _ = batch
        x = x[:, :-1, :, :]  # ignore the angle
        x_hat = self.forward(x)
        loss = nn.functional.mse_loss(x_hat, x)  # Mean Squared Error Loss
        self.log(f"{step}/loss", loss)
        return loss

    def training_step(self, *args, **kwargs):
        return self.shared_step(*args, **kwargs, step="train")

    def validation_step(self, *args, **kwargs):
        return self.shared_step(*args, **kwargs, step="val")

    def test_step(self, *args, **kwargs):
        return self.shared_step(*args, **kwargs, step="test")

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
