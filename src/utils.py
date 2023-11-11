from typing import Union
import pandas as pd
import numpy as np
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


def get_climb_score(climb: torch.Tensor) -> float:
    """Score climb based on validity"""
    climb = climb[:-1, :, :]
    score = 0
    start, middle, finish, foot = climb.sum(dim=[1, 2]).tolist()
    if start in [1, 2]:
        score += 1.0
        argmax = climb[0].argmax()
        x, y = (argmax // climb.size(2), argmax % climb.size(2))
        if y <= 30:
            score += 0.5
    if finish == 1:
        score += 1.0
        argmax = climb[2].argmax()
        x, y = (argmax // climb.size(2), argmax % climb.size(2))
        if y >= 30:
            score += 0.5
    if middle >= 1:
        score += 0.5
    if foot >= 1:
        score += 0.5
    return score


class EncoderDecoder:
    """Converts frames to tensors and back"""

    def __init__(self, holds: pd.DataFrame):
        self.holds = holds
        self.coord_to_id = self._create_coord_to_id()
        self.id_to_coord = self._create_id_to_coord()

    def _create_coord_to_id(self):
        hold_lookup_matrix = np.zeros((48, 48), dtype=int)
        for i in range(48):
            for j in range(48):
                hold = self.holds[
                    (self.holds["x"] == (i * 4 + 4)) & (self.holds["y"] == (j * 4 + 4))
                ]
                if not hold.empty:
                    hold_lookup_matrix[i, j] = int(hold.index[0])
        return hold_lookup_matrix

    def _create_id_to_coord(self):
        id_to_coord = self.holds[["x", "y"]]
        id_to_coord = (id_to_coord - 4) // 4
        return id_to_coord.transpose().to_dict(orient="list")

    def str_to_tensor(self, frames: str, angle: float) -> torch.Tensor:
        angle_matrix = torch.ones((1, 48, 48), dtype=torch.float32) * (angle / 70)
        matrix = torch.zeros((4, 48, 48), dtype=torch.float32)
        for frame in frames.split("p")[1:]:
            hold_id, color = frame.split("r")
            hold_id, color = int(hold_id), int(color) - 12
            coords = self.id_to_coord[hold_id]
            matrix[color, coords[0], coords[1]] = 1
        return torch.cat((matrix, angle_matrix), dim=0)

    def tensor_to_str(self, matrix: torch.Tensor) -> str:
        angle = (matrix[-1].mean() * 70).round().long().item()
        matrix = matrix[:-1, :, :]
        frames = []
        counter = [0,0,0,0]
        for color, x, y in zip(*torch.where(matrix)):
            counter[color] += 1
            color, x, y = color.item(), x.item(), y.item()
            if counter[color] > 2 and color in [0,2]:
                continue
            hold_id = self.coord_to_id[x, y]
            role = color + 12
            frames.append((hold_id, role))
        sorted_frames = sorted(frames, key=lambda x: x[0])
        return (
            "".join([f"p{hold_id}r{role}" for hold_id, role in sorted_frames]), angle
        )

    def __call__(self, *args):
        if len(args) == 1:
            return self.tensor_to_str(*args)
        elif len(args) == 2:
            return self.str_to_tensor(*args)
        else:
            raise ValueError(f"Only 2 input args allowed! You provided {len(args)}")


def jaccard_similarity(frames1: str, frames2: str):
    set1 = set([x[:-3] for x in frames1.split("p")[1:]])
    set2 = set([x[:-3] for x in frames2.split("p")[1:]])
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def combine_handholds(tensor):
    tensor = tensor.long()
    # Combine start, middle, and finish holds into one channel
    handholds = tensor[0] | tensor[1] | tensor[2]  # Logical OR operation
    footholds = tensor[3]
    return torch.stack([handholds, footholds])


def check_nearby(hold1, hold2, threshold=1):
    y1, x1 = torch.where(hold1)
    y2, x2 = torch.where(hold2)

    nearby_count = 0
    for y, x in zip(y1, x1):
        # Check if there's any hold in hold2 within the threshold distance
        if torch.any(
            (torch.abs(y2 - y) <= threshold) & (torch.abs(x2 - x) <= threshold)
        ):
            nearby_count += 1
    maxlen = max(len(y1), len(y2))
    return nearby_count / maxlen if maxlen > 0 else 0


def climb_similarity(climb1, climb2, threshold=1):
    combined_climb1 = combine_handholds(climb1)
    combined_climb2 = combine_handholds(climb2)

    similarity_scores = []
    for i in range(2):  # Iterate over handholds and footholds
        similarity_scores.append(
            check_nearby(combined_climb1[i], combined_climb2[i], threshold=threshold)
        )

    return sum(similarity_scores) / len(similarity_scores)
