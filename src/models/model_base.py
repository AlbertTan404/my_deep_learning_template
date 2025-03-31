from typing import Dict
import lightning.pytorch as pl
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

from ..utils import instantiate_from_config


class ModelBase(pl.LightningModule):
    def __init__(
        self,
        model_kwargs,
        training_kwargs,
        all_config=None,
    ):
        super().__init__()

        self.all_config = all_config
        self.training_kwargs = training_kwargs
        self.model_kwargs = model_kwargs
        self.save_hyperparameters()

    def configure_optimizers(self):
        kwargs = self.all_config.model.training_kwargs
        tuned_parameters = [p for p in self.parameters() if p.requires_grad]

        optimizer = instantiate_from_config(kwargs.optimizer, extra_kwargs={'params': tuned_parameters})

        if not hasattr(kwargs, 'scheduler') or kwargs.scheduler is None:
            return {
                'optimizer': optimizer
            }

        if kwargs.scheduler == 'cosine_schedule_with_warmup':
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=kwargs.warmup_steps, num_training_steps=kwargs.num_training_steps)
        else:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=kwargs.warmup_steps)

        self.lr_scheduler = scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

    def training_step(self, batch, batch_idx):
        log_dict = self.get_log_dict(batch, 'train', batch_idx)
        log_dict.update(self.extra_training_step(batch=batch, batch_idx=batch_idx))
        self.log_dict(log_dict, sync_dist=True, prog_bar=True, batch_size=self.all_config.dataloader.batch_size)
        return log_dict['train/total_loss']

    def validation_step(self, batch, batch_idx):
        log_dict = self.get_log_dict(batch, 'val', batch_idx)
        log_dict.update(self.extra_validation_step(batch=batch, batch_idx=batch_idx))
        if 'monitor' not in log_dict.keys():
            log_dict['monitor'] = - log_dict['val/total_loss']
        self.log_dict(log_dict, sync_dist=True, prog_bar=True, batch_size=self.all_config.dataloader.batch_size)
        return log_dict['val/total_loss']

    def test_step(self, batch, batch_idx=None) -> Dict:
        raise NotImplementedError("Should be implemented in the child class.")

    def get_log_dict(self, batch, split, batch_idx=None) -> Dict:
        raise NotImplementedError("Should be implemented in child class.")

    def extra_training_step(self, batch, batch_idx=None) -> Dict:
        return {}

    def extra_validation_step(self, batch, batch_idx=None) -> Dict:
        return {}
