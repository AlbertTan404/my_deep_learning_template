from torch.utils.data import Dataset
from pathlib import Path


class DatasetBase(Dataset):
    def __init__(
        self,
        data_root,
        split='train',
        epoch_scaling=1,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.epoch_scaling = epoch_scaling

    @property
    def real_length(self):
        raise NotImplementedError

    def __len__(self):
        return self.real_length * self.epoch_scaling

    def __getitem__(self, index):
        index = index % self.real_length
        return self.getitem(index=index)

    def getitem(self, index):
        raise NotImplementedError
