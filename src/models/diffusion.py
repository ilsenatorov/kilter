from typing import Literal

import pytorch_lightning as pl
import torch
from denoising_diffusion_pytorch import GaussianDiffusion, Unet
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, Unet1D
from denoising_diffusion_pytorch.simple_diffusion import GaussianDiffusion as ViTGaussianDiffusion
from denoising_diffusion_pytorch.simple_diffusion import UViT

from .guided_diffusion import GuidedGaussianDiffusion, GuidedUnet


class BaseDiffusionModel(pl.LightningModule):
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        loss = self.diffusion.forward(img)
        return loss

    def training_step(self, batch: list[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, difficulty = batch
        x = x
        loss = self.forward(x)

        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = {
            "monitor": "train/loss",
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                verbose=True,
                factor=0.1,
                patience=5,
                mode="min",
            ),
        }
        return [optimizer], [lr_scheduler]


class GuidedDiffusionModel(BaseDiffusionModel):
    def __init__(self, dim: int, *, timesteps: int = 1000, lr: float = 1e-4, **kwargs):
        super().__init__()
        unet = GuidedUnet(dim=dim, channels=5)
        self.diffusion = GuidedGaussianDiffusion(unet, image_size=48, timesteps=timesteps)
        self.lr = lr

    def forward(self, img: torch.Tensor, difficulty: torch.Tensor):
        loss = self.diffusion.forward(img, classes=difficulty)
        return loss

    def training_step(self, batch):
        x, difficulty = batch
        x = x.float()
        difficulty = difficulty.float()
        loss = self.forward(x, difficulty.unsqueeze(-1))
        self.log("train/loss", loss)
        return loss


class DiffusionModel(BaseDiffusionModel):
    def __init__(
        self,
        dim: int,
        timesteps: int = 1000,
        objective: Literal["pred_noise", "pred_x0", "pred_v"] = "v",
        lr: float = 1e-4,
        **kwargs,
    ):
        super().__init__()
        unet = Unet(dim=dim, channels=5)
        self.diffusion = GaussianDiffusion(unet, image_size=48, timesteps=timesteps, objective=objective)
        self.lr = lr


class SimpleDiffusionModel(BaseDiffusionModel):
    def __init__(self, dim: int, timesteps: int = 1000, objective: Literal["eps", "v"] = "v", lr: float = 1e-4, **kwargs):
        super().__init__()
        unet = UViT(dim=dim, channels=5)
        self.diffusion = ViTGaussianDiffusion(
            unet, image_size=48, num_sample_steps=timesteps, pred_objective=objective, channels=5
        )
        self.lr = lr


class TextDiffusionModel(BaseDiffusionModel):
    def __init__(
        self,
        embedding_dim: int = 64,
        seq_length: int = 64,
        lr: float = 1e-4,
        objective: Literal["pred_noise", "pred_x0", "pred_v"] = "pred_noise",
        **kwargs,
    ):
        super().__init__()
        self.lr = lr
        self.embedding = torch.nn.Embedding(550, embedding_dim, padding_idx=0)
        model = Unet1D(dim=embedding_dim, dim_mults=(1, 2, 4, 8), channels=embedding_dim)
        self.diffusion = GaussianDiffusion1D(
            model, seq_length=seq_length, timesteps=1000, objective=objective, auto_normalize=False
        )

    def forward(self, x: torch.Tensor):
        batch_size, n_tokens = x.size()
        x = self.embedding(x)
        x = x.view(batch_size, -1, n_tokens)
        loss = self.diffusion.forward(x)
        return loss
