import torch
from denoising_diffusion_pytorch.simple_diffusion import UViT
from denoising_diffusion_pytorch.simple_diffusion import (
    GaussianDiffusion as ViTGaussianDiffusion,
)
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import pytorch_lightning as pl
from ..utils import plot_climb, get_climb_score, climb_loss


class DiffusionModel(pl.LightningModule):
    def forward(self, img):
        loss = self.diffusion(img.float())
        return loss

    def training_step(self, batch):
        x, angle, difficulty = batch
        loss = self.forward(x)
        if self.global_step % 500 == 1:
            sampled_images = self.diffusion.sample(12)
            formatted_images = list((sampled_images > 0.5).long())
            self.logger.log_image(
                key="samples", images=[plot_climb(x) for x in formatted_images]
            )
            scores = torch.tensor(
                [get_climb_score(x) for x in formatted_images], dtype=torch.float32
            )
            self.log("train/scores", scores.mean())
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


class DiffusionUNet(DiffusionModel):
    def __init__(self, config):
        super().__init__()
        unet = Unet(dim=config["dim"], dim_mults=(1, 2, 4, 8), channels=4)
        self.diffusion = GaussianDiffusion(
            unet,
            image_size=48,
            timesteps=config["timesteps"],
            objective=config["objective"],
        )
        self.lr = config["lr"]


class DiffusionUViT(DiffusionModel):
    def __init__(self, config):
        super().__init__()
        unet = UViT(dim=config["dim"], dim_mults=(1, 2, 4, 8), channels=4)
        self.diffusion = ViTGaussianDiffusion(
            unet,
            image_size=48,
            num_sample_steps=config["timesteps"],
            channels=4,
            pred_objective=config["objective"],
        )
        self.lr = config["lr"]