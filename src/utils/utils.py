import sys
import copy
import torch
import importlib
from datetime import datetime
from omegaconf.dictconfig import DictConfig


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


def dict_to_device(data: dict, device: torch.device):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.to(device)
    return data


def is_debug_mode():
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None
