import os 
import itertools
import argparse
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

import torch
import lightning.pytorch as pl
from src.utils import instantiate_from_config, get_timestamp, setup_logger


logger = setup_logger(__name__)


def get_train_val_loader(config):
    raise NotImplementedError


def preprocess_config(config, args, unknown_args):
    def set_nested_value(inplace_dict, key_path, value):
        keys = key_path.split('.')
        for key in keys[:-1]:
            inplace_dict = inplace_dict[key]
        inplace_dict[keys[-1]] = value
    
    def expanduser(inplace_dict):
        for k, v in inplace_dict.items():
            if isinstance(v, (dict, DictConfig)):
                expanduser(v)
            elif isinstance(v, str) and k[0] == '~':
                inplace_dict[k] = os.path.expanduser(v)

    # set unknown args to config
    for k, v in itertools.pairwise(unknown_args):
        try:
            v = int(v)
        except:
            pass
        set_nested_value(config, k, v)

    # set project signature
    dataset_name = args.dataset_name
    project_name = config.model.target.split('.')[-2] + '_logs'
    config.trainer.logger.project = project_name
    config.trainer.logger.name = f'{get_timestamp()}-{dataset_name}'

    # devices
    devices = args.devices
    if devices is not None:
        devices = devices.split(',')
        devices = [int(rank) for rank in devices]
        config.trainer.devices = devices
    device_count = torch.cuda.device_count()
    if len(config.trainer.devices) > device_count:
        config.trainer.devices = list(range(device_count))
        logger.warn(f'using {device_count} devices')

    # batch size for ddp
    total_bs = config.dataloader.batch_size
    num_devices = len(config.trainer.devices)
    bs_per_device = total_bs // num_devices
    real_bs = bs_per_device * num_devices
    if real_bs != total_bs:
        logger.warn(f'real batch size is {real_bs}')
    config.dataloader.batch_size = bs_per_device

    # expand all ~ in config to user home path
    expanduser(config)

    return config


def get_processed_args_and_config():
    args, unknown_args = get_args()
    raw_config = OmegaConf.load(f'src/configs/{args.config_name}.yaml')
    OmegaConf.resolve(raw_config)
    config = preprocess_config(raw_config, args, unknown_args)
    return args, config


def get_args():
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument(
        '--config_name',
        default=''
    )

    # training
    parser.add_argument(
        '--devices',
        type=str,
        default='1',
    )

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main():
    args, config = get_processed_args_and_config()
    pl.seed_everything(config.seed)

    model = instantiate_from_config(config.model, extra_kwargs={"all_config": config})

    train_loader, val_loader = get_train_val_loader(config)

    epoch_length = len(train_loader) // len(config.trainer.devices)
    config.model.training_kwargs['num_training_steps'] = epoch_length * config.trainer.max_epochs

    trainer = instantiate_from_config(config.trainer)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
