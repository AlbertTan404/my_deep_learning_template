import os
import shutil
import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig, ListConfig
import torch
import lightning.pytorch as pl

from src.utils import instantiate_from_config, get_timestamp, setup_logger


logger = setup_logger(__name__)


def instantiate_callbacks(callback_configs: ListConfig):
    callbacks = []
    for callback_cfg in callback_configs:
        callbacks.append(instantiate_from_config(callback_cfg))
    
    return callbacks


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
        # dictionary and list are not supported so far.
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
    if args.no_log:
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
        logger.warning(f'real batch size is {real_bs}')
    config.dataloader.batch_size = bs_per_device

    # epoch scaling
    epoch_scaling = config.dataset.get('epoch_scaling')
    if epoch_scaling is not None and epoch_scaling != 1:
        config.trainer.max_epochs = int(config.trainer.max_epochs / epoch_scaling)
        logger.info(f'Training epoch length is scaled by {epoch_scaling}, thus the num of epochs is decreased to {config.trainer.max_epochs}')
    
    # customize anything here
    config = preprocess_config_hook(config)

    logger.info(f'running with config: {config}')
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

    # merge args into config
    config = OmegaConf.merge(config, OmegaConf.create({'args': vars(args)}))
    
    return args, config


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        type=str,
        default='toy_model'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='toy_dataset'
    )

    parser.add_argument(
        '--trainer',
        type=str,
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
        type=str,
        help='add suffix to log dir',
        default=''
    )

    parser.add_argument(
        '--resume_ckpt_path',
        type=str,
        help='resume training from ckpt',
        default=None
    )

    parser.add_argument(
        '--load_ckpt_path',
        type=str,
        help='load ckpt as initialization',
        default=None
    )

    parser.add_argument(
        '--workspace_path',
        type=str,
        help='assign the path of user workspace directory',
        default='~'
    )

    parser.add_argument(
        '--do_test',
        help='test after training',
        action='store_true'
    )

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main():
    args, config = get_processed_args_and_config()

    pl.seed_everything(config.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    data_module: pl.LightningDataModule = instantiate_from_config(config.data_module, extra_kwargs={'all_config': config})

    model: pl.LightningModule = instantiate_from_config(config.model, extra_kwargs={"all_config": config})
    if p := args.load_ckpt_path:
        model.load_state_dict(torch.load(p, map_location='cpu')['state_dict'])

    trainer: pl.Trainer = instantiate_from_config(config.trainer, extra_kwargs={'callbacks': instantiate_callbacks(config.callbacks)})

    try:
        try:
            if trainer.global_rank == 0:
                shutil.copytree('src', os.path.join(trainer.logger.log_dir, 'src_backup'))  # backup src directory
        except: pass

        trainer.fit(model=model, datamodule=data_module, ckpt_path=args.resume_ckpt_path)

        if args.do_test:
            # evaluation
            results = trainer.test(ckpt_path='best')[0]  # the first dataloader
            logger.info(f'evaluation results: {results}')

        if trainer.global_rank == 0:
            shutil.move(trainer.logger.log_dir, trainer.logger.log_dir + '_trained')

    except Exception as e:
        raise e
    else:
        # mark log dir as trained
        if trainer.global_rank == 0:
            shutil.move(trainer.logger.log_dir, trainer.logger.log_dir + '_trained' + '_tested')


if __name__ == '__main__':
    main()
