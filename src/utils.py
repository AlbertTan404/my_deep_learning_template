import logging
import importlib
from datetime import datetime
from omegaconf.dictconfig import DictConfig


def setup_logger(name, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    logger.addHandler(console_handler)
    if log_file:
        logger.addHandler(file_handler)

    return logger


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
        if config_dict == '__is_first_stage__':
            return None
        elif config_dict == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    target_kwargs = dict(config_dict.get('kwargs', dict()))

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
