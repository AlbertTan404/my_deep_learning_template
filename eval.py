import argparse
import shutil
import tqdm
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import torch
import pprint
from torch.utils.data import DataLoader, ConcatDataset

from src.utils.utils import get_obj_from_str, instantiate_from_config, setup_logger


logger = None


@torch.no_grad()
def evaluate(args, model, test_dataloader):
    logger.info(f'result: {None}')


def get_args_and_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='logs/model_name/dataset_name/signature_trained/checkpoints/best.ckpt'
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

    args.ckpt_path = Path(args.ckpt_path)
    log_dir = args.ckpt_path.parent.parent

    args.device = torch.device(args.device)

    args.log_file_path = str((log_dir / 'results.log').expanduser())

    config = OmegaConf.load(log_dir / 'hparams.yaml').all_config

    return args, config


def main():
    global logger
    args, config = get_args_and_config()

    logger = setup_logger(__file__, log_file=args.log_file_path)

    logger.info(f'\n-----------------------------------------------\n')
    logger.info(f'Evaluating with ckpt: {args.ckpt_path}')
    logger.info(f'Evaluation config: {pprint.pformat(config)}')

    model_cls = get_obj_from_str(config.model.target)
    model = model_cls.load_from_checkpoint(str(args.ckpt_path), map_location=args.device).eval()

    # load val data
    test_dataset = instantiate_from_config(config.dataset, extra_kwargs={'split': 'test'})

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, persistent_workers=True, drop_last=True)

    evaluate(args, model, test_dataloader)


if __name__ == '__main__':
    main()
