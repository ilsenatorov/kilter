import torch
from .guided_diffusion import GuidedGaussianDiffusion, GuidedUnet
import pytorch_lightning as pl
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.simple_diffusion import UViT
from denoising_diffusion_pytorch.simple_diffusion import GaussianDiffusion as ViTGaussianDiffusion


def get_caption(x, grade):
    angle = (x[-1].mean() * 70).round().long().item()
    return f"Grade: {grade:.2f}, Angle: {angle}"


class BaseDiffusionModel(pl.LightningModule):

    def forward(self, img):
        loss = self.diffusion.forward(img)
        return loss

    def training_step(self, batch):
        x, difficulty = batch
        x = x.float()
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
    def __init__(self, dim:int, timesteps:int=1000, lr:float=1e-4, **kwargs):
        super().__init__()
        unet = GuidedUnet(dim=dim, channels=5)
        self.diffusion = GuidedGaussianDiffusion(unet, image_size=48, timesteps=timesteps)
        self.lr = lr

    def forward(self, img, difficulty):
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
    def __init__(self, dim:int, timesteps:int=1000, objective:str="v", lr:float=1e-4, **kwargs):
        super().__init__()
        unet = Unet(dim=dim, channels=5)
        self.diffusion = GaussianDiffusion(unet, image_size=48, timesteps=timesteps, objective=objective)
        self.lr = lr

class SimpleDiffusionModel(BaseDiffusionModel):
    def __init__(self, dim:int, timesteps:int=1000, objective:str="v", lr:float=1e-4, **kwargs):
        super().__init__()
        unet = UViT(dim=dim, channels=5)
        self.diffusion = ViTGaussianDiffusion(unet, image_size=48, num_sample_steps=timesteps, pred_objective=objective, channels=5)
        self.lr = lr