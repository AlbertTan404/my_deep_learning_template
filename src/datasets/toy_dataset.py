import torch
from .dataset_base import DataModuleBase, DatasetBase


class ToyDataset(DatasetBase):
    def __init__(self, dataset_dir=None, split='train', epoch_scaling=1, tiny_dataset=False):
        super().__init__(dataset_dir, split, epoch_scaling, tiny_dataset)
        self._real_length = 1000
    
    @property
    def real_length(self):
        return self._real_length
    
    def getitem(self, index):
        x = torch.randn(size=(128, ))
        y = x.mean()
        return {
            'x': x,
            'y': y
        }


class ToyDataModule(DataModuleBase):
    def __init__(self, all_config=None, **kwargs):
        super().__init__(all_config, **kwargs)
