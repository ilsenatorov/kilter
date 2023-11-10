import torch
from denoising_diffusion_pytorch.simple_diffusion import UViT, GaussianDiffusion
import pytorch_lightning as pl
from ..utils import plot_climb, get_climb_score, climb_similarity


class DiffusionUNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        unet = UViT(dim=config["dim"], dim_mults=(1, 2, 4, 8), channels=4)
        self.diffusion = GaussianDiffusion(unet, image_size=48, channels=4)
        self.lr = config["lr"]

    def forward(self, img):
        loss = self.diffusion(img.float())
        return loss

    def training_step(self, batch):
        x, angle, difficulty = batch
        loss = self.forward(x)
        if self.global_step % 500 == 1:
            sampled_images = self.diffusion.sample(batch_size=x.size(0))
            formatted_images = list((sampled_images > 0.5).long())
            self.logger.log_image(key="samples", images=[plot_climb(x) for x in formatted_images[:4]])
            scores = torch.tensor([get_climb_score(x) for x in formatted_images], dtype=torch.float32)
            self.log("train/score", scores.mean())
            self.log("train/hist_dist", (x.mean(0) - sampled_images.mean(0)).abs().mean())
            scores = []
            for i in list(x.long()):
                for j in formatted_images:
                    scores.append(climb_similarity(i, j))
            self.log("train/climb_sim", torch.tensor(scores).mean())
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
