import torch
from .guided_diffusion import GuidedGaussianDiffusion, GuidedUnet
import pytorch_lightning as pl
from ..utils import plot_climb, get_climb_score
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.simple_diffusion import UViT
from denoising_diffusion_pytorch.simple_diffusion import GaussianDiffusion as ViTGaussianDiffusion


def get_caption(x, grade):
    angle = (x[-1].mean() * 70).round().long().item()
    return f"Grade: {grade:.2f}, Angle: {angle}"

class GuidedDiffusionModel(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        unet = GuidedUnet(dim=config["dim"], dim_mults=(1, 2, 4, 8), channels=5)
        self.diffusion = GuidedGaussianDiffusion(unet, image_size=48, timesteps=config["timesteps"])
        self.lr = config["lr"]

    def forward(self, img, difficulty):
        loss = self.diffusion.forward(img, classes=difficulty)
        return loss

    def training_step(self, batch):
        x, difficulty = batch
        x = x.float()
        difficulty = difficulty.float()
        loss = self.forward(x, difficulty.unsqueeze(-1))
        if self.global_step % 2000 == 1:
            grades = torch.randint(10, 30, (difficulty.size(0), 1), dtype=torch.float32, device=self.device)
            sampled_images = self.diffusion.sample(grades)
            formatted_images = list((sampled_images > 0.5).long())
            captions = []
            for climb, grade in zip(list(sampled_images), grades.view(-1).tolist()):
                captions.append(get_caption(climb, grade))
            self.logger.log_image(
                key="samples", images=[plot_climb(x) for x in formatted_images], caption=captions,
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
    
class BaseDiffusionModel(pl.LightningModule):

    def forward(self, img, difficulty):
        loss = self.diffusion.forward(img, classes=difficulty)
        return loss

    def training_step(self, batch):
        x, difficulty = batch
        x = x.float()
        loss = self.forward(x)
        if self.global_step % 2000 == 1:
            sampled_images = self.diffusion.sample(25)
            formatted_images = list((sampled_images > 0.5).long())
            captions = []
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
    

class DiffusionModel(BaseDiffusionModel):
    def __init__(self, config:dict):
        super().__init__()
        unet = Unet(dim=config["dim"], channels=5)
        self.diffusion = GaussianDiffusion(unet, image_size=48, timesteps=config["timesteps"], objective=config["objective"], )

class DiffusionModel(BaseDiffusionModel):
    def __init__(self, config:dict):
        super().__init__()
        unet = UViT(dim=config["dim"], channels=5, vit_dropout=config["dropout"])
        self.diffusion = ViTGaussianDiffusion(unet, image_size=48, num_sample_steps=config["timesteps"], pred_objective=config["objective"])