import torch
from PIL import Image

colors = torch.tensor(
    [
        [0, 255, 0],  # Green
        [0, 0, 255],  # Blue
        [128, 0, 128],  # Purple
        [255, 165, 0],  # Orange
    ],
    dtype=torch.uint8,
)


def plot_climb(climb: torch.Tensor) -> Image:
    colored_image = torch.zeros((48, 48, 3), dtype=torch.uint8)
    for i in range(4):
        mask = climb[i].to(torch.bool)
        colored_image[mask] = colors[i]
    return Image.fromarray(colored_image.numpy()).rotate(90)
