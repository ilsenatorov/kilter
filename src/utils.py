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


def get_climb_score(climb):
    """Score climb based on validity"""
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


def encode_frame(frames: str, holds) -> torch.Tensor:
    matrix = torch.zeros((4, 48, 48), dtype=torch.long)
    for frame in frames.split("p")[1:]:
        hold_id, color = frame.split("r")
        hold_id, color = int(hold_id), int(color) - 12
        hold = holds.loc[hold_id]
        x, y = (hold.x - 4) // 4, (hold.y - 4) // 4
        matrix[color, x, y] = 1
    return matrix.float()


def decode_frame(matrix, holds) -> str:
    frames = []
    for color in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            for y in range(matrix.shape[2]):
                if matrix[color, x, y] == 1:
                    hold_id = holds[(holds["x"] == (x * 4 + 4)) & (holds["y"] == (y * 4 + 4))].index[0]
                    role = color + 12
                    frames.append((hold_id, role))
    sorted_frames = sorted(frames, key=lambda x: x[0])
    return "".join([f"p{hold_id}r{role}" for hold_id, role in sorted_frames])


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
        if torch.any((torch.abs(y2 - y) <= threshold) & (torch.abs(x2 - x) <= threshold)):
            nearby_count += 1
    maxlen = max(len(y1), len(y2))
    return nearby_count / maxlen if maxlen > 0 else 0


def climb_similarity(climb1, climb2, threshold=1):
    combined_climb1 = combine_handholds(climb1)
    combined_climb2 = combine_handholds(climb2)

    similarity_scores = []
    for i in range(2):  # Iterate over handholds and footholds
        similarity_scores.append(check_nearby(combined_climb1[i], combined_climb2[i], threshold=threshold))

    return sum(similarity_scores) / len(similarity_scores)
