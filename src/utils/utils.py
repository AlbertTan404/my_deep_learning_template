import sys
from typing import Dict, Any
import copy
import importlib
from pathlib import Path
from datetime import datetime
from omegaconf.dictconfig import DictConfig
import numpy as np
import torch


def get_timestamp():
    return datetime.now().strftime('%Y%m%d-%H%M%S')


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, extra_kwargs=dict()):
    config_dict = dict(config)
    if not "target" in config_dict:
        raise ValueError(f'target not found in {config}')

    target_kwargs = copy.deepcopy(config_dict)
    target_kwargs.pop('target')

    for k, v in target_kwargs.items():
        if isinstance(v, DictConfig) and 'target' in v.keys():
            target_kwargs[k] = instantiate_from_config(v)
    target_kwargs.update(extra_kwargs)

    return get_obj_from_str(config_dict["target"])(**target_kwargs)


def dict_apply(x, func):
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def dict_to_device(d: Dict[str, Any], device):
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(device=device)
    return d


def list_subdirs(path: Path):
    return [d for d in path.glob('*') if not d.is_file()]


def is_debug_mode():
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


def get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval
