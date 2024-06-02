import argparse
import shutil
import tqdm
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, ConcatDataset

from src.utils.utils import get_obj_from_str, instantiate_from_config, setup_logger


logger = None


@torch.no_grad()
def evaluation(*args, **kwargs):
    logger.info(f'result: {None}')


def get_args_and_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_dir',
        type=str,
        default='logs/model_name/dataset_name/signature_trained'
    )

    parser.add_argument(
        '--device',
        type=int,
        default=1
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32
    )

    args = parser.parse_args()
    args.model_dir = Path(args.model_dir)
    args.device = torch.device(args.device)
    args.log_file_path = str((args.log_dir / 'results.log').expanduser())

    config = OmegaConf.load(args.model_dir / 'hparams.yaml').all_config

    return args, config


def main():
    global logger
    args, config = get_args_and_config()

    logger = setup_logger(__file__, log_file=args.log_file_path)
    logger.info(f'results will be saved to {args.log_file_path}')

    logger.info(f'\nEvaluation config: {config}')

    model_cls = get_obj_from_str(config.model.target)
    model = model_cls.load_from_checkpoint(list((args.model_dir / 'checkpoints').glob('*.ckpt'))[-1], map_location=args.device).eval()  # load last checkpoint (maybe the only checkpoint)

    # load val data
    if len(config.dataset) > 1:
        test_ds_list = []
        for dataset_config in config.dataset:
            test_ds_list.append(instantiate_from_config(dataset_config, extra_kwargs={'split': 'test'}))
        test_dataset = ConcatDataset(test_ds_list)
    else:
        test_dataset = instantiate_from_config(config.dataset[0], extra_kwargs={'split': 'test'})

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, persistent_workers=True, drop_last=True)

    evaluation(model, test_dataloader)


if __name__ == '__main__':
    main()
