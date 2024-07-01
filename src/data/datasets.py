from pathlib import Path
from typing import Callable, Literal, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils import EncoderDecoder, Tokenizer


class KilterDataset(Dataset):
    """Kilter dataset for prediction, use split to select between train, val and test"""

    def __init__(
        self,
        root_dir: str = "data/kilter",
        split: Literal["train", "val", "test"] = "train",
        transform: Callable = None,
    ):
        self.root_dir = Path(root_dir)
        self.climbs = pd.read_csv(self.root_dir / f"raw/{split}.csv", index_col=0)
        self.encdec = EncoderDecoder()
        self.transform = transform
        self.split = split
        self.data = self._load_or_preprocess_data()

    def _load_or_preprocess_data(self):
        processed_file = Path(self.root_dir) / f"processed/{self.split}.pt"
        if processed_file.exists():
            data = torch.load(processed_file)
        else:
            data = self._preprocess_data()
            (self.root_dir / "processed").mkdir(parents=True, exist_ok=True)
            torch.save(data, processed_file)
        return data

    def _preprocess_data(self):
        data = []
        for _, row in tqdm(self.climbs.iterrows()):
            img = self.encdec(row["frames"], float(row["angle"]))
            difficulty = float(row["difficulty_average"])
            data.append((img, difficulty))
        return data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        image, difficulty = self.data[idx]

        if self.transform:
            image = self.transform(image)

        return image, difficulty

    def __len__(self):
        return len(self.data)


class KilterDiffusionDataset(KilterDataset):
    """Diffusion dataset, has no splits and loads all of the data."""

    def __init__(
        self,
        root_dir: str = "data/kilterdiffuse",
        transform: Callable = None,
    ):
        self.root_dir = Path(root_dir)
        self.climbs = pd.read_csv(self.root_dir / "raw/all_climbs.csv", index_col=0)
        self.encdec = EncoderDecoder()
        self.transform = transform
        self.data = self._load_or_preprocess_data()

    def _load_or_preprocess_data(self):
        processed_file = Path(self.root_dir) / f"processed/diffusion.pt"
        if processed_file.exists():
            data = torch.load(processed_file)
        else:
            data = self._preprocess_data()
            (self.root_dir / "processed").mkdir(parents=True, exist_ok=True)
            torch.save(data, processed_file)
        return data


class KilterTextDiffusionDataset(Dataset):
    """Kilter dataset for prediction, use split to select between train, val and test"""

    def __init__(self, root_dir: str = "data/kiltertextdiffuse"):
        self.root_dir = Path(root_dir)
        self.climbs = pd.read_csv(self.root_dir / "raw/all_climbs.csv", index_col=0)
        self.tokenizer = Tokenizer(self.climbs)
        self.data = self._load_or_preprocess_data()

    def _load_or_preprocess_data(self):
        processed_file = Path(self.root_dir) / f"processed/diffusion.pt"
        if processed_file.exists():
            data = torch.load(processed_file)
        else:
            data = self._preprocess_data()
            (self.root_dir / "processed").mkdir(parents=True, exist_ok=True)
            torch.save(data, processed_file)
        return data

    def _preprocess_data(self):
        data = []
        for _, row in tqdm(self.climbs.iterrows()):
            x = self.tokenizer(row["frames"])
            difficulty = float(row["difficulty_average"])
            data.append((x, difficulty))
        return data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        x, difficulty = self.data[idx]
        return x, difficulty

    def __len__(self):
        return len(self.data)
