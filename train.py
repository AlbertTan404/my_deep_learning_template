import os
from typing import List
import shutil
import pprint
import argparse
from omegaconf import OmegaConf, DictConfig, ListConfig
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from src.utils import instantiate_from_config, get_timestamp, setup_logger, is_debug_mode


logger = setup_logger(__name__)


def instantiate_callbacks(callback_configs: ListConfig):
    callbacks = []
    for callback_cfg in callback_configs:
        callbacks.append(instantiate_from_config(callback_cfg))
    
    return callbacks


def get_train_val_dataloader(config):

    train_ds = instantiate_from_config(config.dataset, extra_kwargs={'split': 'train'})
    val_ds = instantiate_from_config(config.dataset, extra_kwargs={'split': 'val'})

    train_dataloader = DataLoader(train_ds, **config.dataloader, shuffle=True)
    val_dataloader = DataLoader(val_ds, **config.dataloader, shuffle=False)

    return train_dataloader, val_dataloader


def _preprocess_config(config, args, unknown_args):
    def set_config_key_value(inplace_dict, key_path, value):
        flag = False
        def bfs_set_config_key_value(inplace_dict, key, value):
            nonlocal flag
            if key in inplace_dict.keys():
                inplace_dict[key] = value
                flag = True
            for v in inplace_dict.values():
                if isinstance(v, (DictConfig, dict)):
                    bfs_set_config_key_value(inplace_dict=v, key=key, value=value)
                elif isinstance(v, ListConfig):
                    for item in v:
                        bfs_set_config_key_value(inplace_dict=item, key=key, value=value)
        
        keys = key_path.split('.')  # dataset.a.b = 1
        len_keys = len(keys)
        if len_keys == 1:
            bfs_set_config_key_value(inplace_dict, key=key_path, value=value)
            if flag:
                return
            else:
                raise ValueError(f'{key_path} is not found in config')

        for key_idx in range(len_keys - 1):  # 
            inplace_dict = inplace_dict[keys[key_idx]]

            if isinstance(inplace_dict, ListConfig):
                for item in inplace_dict:
                    for sub_key_idx in range(key_idx + 1, len_keys - 1):
                        item = item[keys[sub_key_idx]]
                    item[keys[-1]] = value
                return

        inplace_dict[keys[-1]] = value

    # set unknown args to config
    for unknown in unknown_args:
        k, v = unknown.split('=')
        try:
            v = int(v)  # maybe int has the highest priority
        except:
            try:
                v = float(v)
            except:
                # v = bool(v) is not supported as bool('False') -> True
                if (vlower := v.lower()) == 'true':
                    v = True
                elif vlower == 'false':
                    v = False
                elif vlower == 'none':
                    v = None
                # else v = v, the str itself
        set_config_key_value(config, k, v)

    # devices
    devices = args.devices
    if devices is not None:
        config.trainer.devices = [int(rank) for rank in devices.split(',')]

    # set project name and signature for logging
    if args.no_log or is_debug_mode():
        config.trainer.logger = False
    else:
        config.trainer.logger.save_dir = f'logs/{args.model}'
        config.trainer.logger.name = f'{args.dataset}'
        config.trainer.logger.version = get_timestamp() + (f'_{args.log_suffix}' if args.log_suffix != '' else '')

    # batch size for ddp
    total_bs = config.dataloader.batch_size
    num_devices = len(config.trainer.devices)
    bs_per_device = total_bs // num_devices
    real_bs = bs_per_device * num_devices
    if real_bs != total_bs:
        logger.warn(f'real batch size is {real_bs}')
    config.dataloader.batch_size = bs_per_device

    # epoch scaling
    epoch_scaling = config.dataset.get('epoch_scaling')
    if epoch_scaling is not None and epoch_scaling != 1:
        config.trainer.max_epochs = int(config.trainer.max_epochs / epoch_scaling)
        logger.info(f'Training epoch length is scaled by {epoch_scaling}, thus the num of epochs is decreased to {config.trainer.max_epochs}')
    
    # personal design
    config = preprocess_config_hook(config)

    logger.info(f'running with config: {pprint.pformat(config)}')
    return config


def preprocess_config_hook(config):
    return config


def get_processed_args_and_config():
    args, unknown_args = get_args()

    OmegaConf.register_new_resolver("eval", eval)

    # load trainer config
    trainer_config = OmegaConf.load(f'src/configs/trainer/{args.trainer}.yaml')
    OmegaConf.resolve(trainer_config)

    # load model config
    model_config = OmegaConf.load(f'src/configs/models/{args.model}.yaml')
    OmegaConf.resolve(model_config)
    config = OmegaConf.merge(trainer_config, model_config)

    # load dataset config
    dataset_config = OmegaConf.load(f'src/configs/datasets/{args.dataset}.yaml')
    OmegaConf.resolve(dataset_config)
    config = OmegaConf.merge(config, DictConfig(dataset_config))

    config = _preprocess_config(config, args, unknown_args)
    
    return args, config


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        default='my_model'
    )

    parser.add_argument(
        '--dataset',
        default='my_dataset'
    )

    parser.add_argument(
        '--trainer',
        default='default'
    )

    parser.add_argument(
        '--devices',
        type=str,
        default='1',
    )

    parser.add_argument(
        '--no_log',
        help='disable training log',
        action='store_true'
    )

    parser.add_argument(
        '--log_suffix',
        help='add suffix to log dir',
        default=''
    )

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main():
    args, config = get_processed_args_and_config()
    pl.seed_everything(config.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_dataloader, val_dataloader = get_train_val_dataloader(config)
    epoch_length = len(train_dataloader) // len(config.trainer.devices)
    config.model.training_kwargs['num_training_steps'] = epoch_length * config.trainer.max_epochs

    model: pl.LightningModule = instantiate_from_config(config.model, extra_kwargs={"all_config": config})

    trainer: pl.Trainer = instantiate_from_config(config.trainer, extra_kwargs={'callbacks': instantiate_callbacks(config.callbacks)})

    try:
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    except Exception as e:
        raise e
    else:
        # mark log dir as trained
        if trainer.global_rank == 0:
            shutil.move(trainer.logger.log_dir, trainer.logger.log_dir + '_trained')


if __name__ == '__main__':
    main()
