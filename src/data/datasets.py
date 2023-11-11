from torch.utils.data import Dataset
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ..utils import EncoderDecoder


class KilterDataset(Dataset):
    def __init__(
        self,
        root_dir: str = "data",
        split: str = "train",
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        assert split in [
            "train",
            "val",
            "test",
        ], "split needs to be one of train, val or test"
        self.climbs = pd.read_csv(self.root_dir / f"raw/{split}.csv", index_col=0)
        holds = pd.read_csv(self.root_dir / "raw/holds.csv", index_col=0)
        self.encdec = EncoderDecoder(holds)
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

    def __getitem__(self, idx):
        image, difficulty = self.data[idx]

        if self.transform:
            image = self.transform(image)

        return image, difficulty

    def __len__(self):
        return len(self.data)


class KilterDiffusionDataset(KilterDataset):
    def __init__(
        self,
        root_dir: str = "data",
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.climbs = pd.read_csv(self.root_dir / "raw/all_climbs.csv", index_col=0)
        # self.climbs.drop_duplicates("frames", keep="first", inplace=True)
        holds = pd.read_csv(self.root_dir / "raw/holds.csv", index_col=0)
        self.encdec = EncoderDecoder(holds)
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
