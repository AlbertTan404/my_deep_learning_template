import os 
import itertools
import argparse
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

import torch
import lightning.pytorch as pl
from src.utils import instantiate_from_config, get_timestamp, setup_logger, is_debug_mode


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
            elif isinstance(v, str) and v[0] == '~':
                inplace_dict[k] = os.path.expanduser(v)

    # set unknown args to config
    for k, v in itertools.pairwise(unknown_args):
        try:
            v = int(v)
        except:
            try:
                v = float(v)
            except:
                # v = bool(v) is not supported as bool('False') -> True hahaha
                if v.lower() == 'true':
                    v = True
                elif v.lower() == 'false':
                    v = False
                # else v = v, the str itself
        set_nested_value(config, k, v)

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

    # set project name and signature for logging
    if args.no_log or is_debug_mode():
        # don't debug with wandb to upload garbages
        config.trainer.pop('logger')
    else:
        config.trainer.logger.save_dir = f'logs/{config.model.target.split(".")[-1]}'
        config.trainer.logger.name = f'{config.dataset.target.split(".")[-1]}'
        config.trainer.logger.version = get_timestamp()

    # batch size for ddp
    total_bs = config.dataloader.batch_size
    num_devices = len(config.trainer.devices)
    bs_per_device = total_bs // num_devices
    real_bs = bs_per_device * num_devices
    if real_bs != total_bs:
        logger.warn(f'real batch size is {real_bs}')
    config.dataloader.batch_size = bs_per_device

    # link training config and dataset config
    epoch_scaling = config.dataset.get('epoch_scaling')
    if epoch_scaling is not None or epoch_scaling != 1:
        config.trainer.max_epochs = int(config.trainer.max_epochs / epoch_scaling)
        logger.info(f'Epoch length is scaled by {epoch_scaling}, thus the num of epochs is decreased to {config.trainer.max_epochs}')

    # expand all ~ in config to user home path
    # DO NOT expand here to avoid ckpt saving absolute path
    # expand when neccesary
    # expanduser(config)

    return config


def get_processed_args_and_config():
    args, unknown_args = get_args()

    # load model config
    model_config = OmegaConf.load(f'src/configs/models/{args.model}.yaml')
    OmegaConf.resolve(model_config)

    # load dataset config
    dataset_config = OmegaConf.load(f'src/configs/datasets/{args.dataset}.yaml')
    OmegaConf.resolve(dataset_config)

    config = preprocess_config(OmegaConf.merge(model_config, dataset_config), args, unknown_args)
    
    return args, config


def get_args():
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument(
        '--model',
        default='vanilla'
    )

    parser.add_argument(
        '--dataset',
        default='my_dataset'
    )

    # training
    parser.add_argument(
        '--devices',
        type=str,
        default='1',
    )

    parser.add_argument(
        '--no_log',
        action='store_true'
    )

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main():
    args, config = get_processed_args_and_config()
    pl.seed_everything(config.seed)

    train_loader, val_loader = get_train_val_loader(config)

    epoch_length = len(train_loader) // len(config.trainer.devices)
    config.model.training_kwargs['num_training_steps'] = epoch_length * config.trainer.max_epochs

    model = instantiate_from_config(config.model, extra_kwargs={"all_config": config})

    trainer = instantiate_from_config(config.trainer)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
