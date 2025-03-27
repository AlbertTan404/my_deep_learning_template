from torch.utils.data import Dataset
from pathlib import Path


class DatasetBase(Dataset):
    def __init__(
        self,
        data_root,
        split='train',
        epoch_scaling=1,
        tiny_dataset=False
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.epoch_scaling = epoch_scaling
        self.tiny_dataset = tiny_dataset  # recommended to set to ``True`` when debugging

    @property
    def real_length(self):
        raise NotImplementedError

    def __len__(self):
        if self.split == 'train':
            return self.real_length * self.epoch_scaling
        else:
            return self.real_length

    def __getitem__(self, index):
        index = index % self.real_length
        return self.getitem(index=index)

    def getitem(self, index):
        raise NotImplementedError("Should be implemented in the child class.")
