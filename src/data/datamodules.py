import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .datasets import KilterDataset


class KilterDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, num_workers: int, shuffle=True, transform=None):
        super(KilterDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.transform = transform

    def setup(self, stage=None):
        # Define your datasets
        self.train = KilterDataset(split="train", transform=self.transform)
        self.val = KilterDataset(split="val", transform=self.transform)
        self.test = KilterDataset(split="test", transform=self.transform)

    def dataloader(self, dataset, shuffle: bool = True):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self.dataloader(self.train, self.shuffle)

    def val_dataloader(self):
        return self.dataloader(self.val, False)

    def test_dataloader(self):
        return self.dataloader(self.test, False)
