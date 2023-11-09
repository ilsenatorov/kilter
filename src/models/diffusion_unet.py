import torch
from denoising_diffusion_pytorch.simple_diffusion import UViT, GaussianDiffusion
import pytorch_lightning as pl
from ..utils.vis import plot_climb


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
            sampled_images = self.diffusion.sample(batch_size=4)
            self.logger.log_image(
                key="samples",
                images=[plot_climb((x > 0.5).long()) for x in list(sampled_images)],
            )
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
