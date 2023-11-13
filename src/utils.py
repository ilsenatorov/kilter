from typing import Union
import pandas as pd
import numpy as np
import torch
from PIL import Image
import cv2

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
        if y <= 25:
            score += 0.5
    if finish == 1:
        score += 1.0
        argmax = climb[2].argmax()
        x, y = (argmax // climb.size(2), argmax % climb.size(2))
        if y >= 25:
            score += 0.5
    if middle >= 1:
        score += 0.5
    if foot >= 1:
        score += 0.5
    return score


class EncoderDecoder:
    """Converts frames to tensors and back.
    If given tensor - returns string and angle.
    If given string and angle - returns (5,48,48) tensor.
    """

    def __init__(self):
        holds = pd.read_csv("data/raw/holds.csv", index_col=0)
        image_coords = pd.read_csv("figs/image_coords.csv", index_col=0)
        self.coord_to_id = self._create_coord_to_id(holds)
        self.id_to_coord = self._create_id_to_coord(holds)
        self.image_coords = self._create_image_coords(image_coords)

    def _create_coord_to_id(self, holds:pd.DataFrame):
        hold_lookup_matrix = np.zeros((48, 48), dtype=int)
        for i in range(48):
            for j in range(48):
                hold = holds[
                    (holds["x"] == (i * 4 + 4)) & (holds["y"] == (j * 4 + 4))
                ]
                if not hold.empty:
                    hold_lookup_matrix[i, j] = int(hold.index[0])
        return hold_lookup_matrix

    def _create_id_to_coord(self, holds):
        id_to_coord = holds[["x", "y"]]
        id_to_coord = (id_to_coord - 4) // 4
        return id_to_coord.transpose().to_dict(orient="list")

    def _create_image_coords(self, image_coords):
        return {name: (row["x"], row["y"]) for name, row in image_coords.iterrows()}

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
        angle = ((matrix[-1].mean() * 70 / 5).round() * 5).long().item()
        matrix = matrix[:-1, :, :].round().long()
        frames = []
        counter = [0,0,0,0]
        for color, x, y in zip(*torch.where(matrix)):
            counter[color] += 1
            color, x, y = color.item(), x.item(), y.item()
            # too many start/end holds
            if counter[color] > 2 and color in [0,2]: 
                continue
            hold_id = self.coord_to_id[x, y]
            # wrong hold position
            if hold_id == 0:
                continue
            role = color + 12
            frames.append((hold_id, role))
        sorted_frames = sorted(frames, key=lambda x: x[0])
        return (
            "".join([f"p{hold_id}r{role}" for hold_id, role in sorted_frames]), angle
        )
    
    def plot_climb(self, frames:str):
        assert isinstance(frames, str), f"Input must be frames! Got {type(frames)}"
        board_path = "figs/full_board_commercial.png"
        board_image = cv2.imread(board_path)
        for hold in frames.split("p")[1:]:
            hold_id,hold_type = hold.split("r")
            if int(hold_id) not in self.image_coords:
                continue
            radius = 30
            thickness = 2
            if hold_type == str(12):
                color = (0,255,0) #start
            if hold_type == str(13): # hands
                color = (0,200,255)
            if hold_type == str(14): # end
                color = (255,0,255)
            if hold_type == str(15): # feet
                color = (255,165,0)
            image = cv2.circle(board_image, self.image_coords[int(hold_id)], radius, color, thickness)
        return image

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
