from pathlib import Path
import copy
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from ..utils.utils import instantiate_from_config


def get_dataloaders(config):
    train_ds = instantiate_from_config(config.dataset, extra_kwargs={'split': 'train'})
    val_ds = instantiate_from_config(config.dataset, extra_kwargs={'split': 'val'})

    dataloader_config = copy.copy(config.dataloader)
    val_batch_size = dataloader_config.pop('val_batch_size', dataloader_config.batch_size)
    train_dataloader = DataLoader(train_ds, **dataloader_config, shuffle=True, drop_last=True)
    dataloader_config.batch_size = val_batch_size
    val_dataloader = DataLoader(val_ds, **dataloader_config, shuffle=False, drop_last=True)

    try:
        test_ds = instantiate_from_config(config.dataset, extra_kwargs={'split': 'test'})
        test_dataloader = DataLoader(test_ds, **dataloader_config, shuffle=False, drop_last=True)
    except:
        test_dataloader = None

    return train_dataloader, val_dataloader, test_dataloader


class DataModuleBase(pl.LightningDataModule):
    def __init__(
        self,
        data_module_kwargs={},
        all_config={},
    ):
        config = all_config
        super().__init__()
        self._train_dataloader, self._val_dataloader, self._test_dataloader = get_dataloaders(config)
        epoch_length = len(self.train_dataloader) // len(config.trainer.devices)
        config.model.training_kwargs['num_training_steps'] = epoch_length * config.trainer.max_epochs
    
    def train_dataloader(self):
        return self._train_dataloader
    
    def val_dataloader(self):
        return self._val_dataloader
    
    def test_dataloader(self):
        return self._test_dataloader


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
