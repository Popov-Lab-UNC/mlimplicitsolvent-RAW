import os
from omegaconf import OmegaConf

def load_config(path):
    """ Loads configs/default.yaml, and overrides any values
    with those in the file at `path` if it exists. """

    cfg = OmegaConf.load("configs/default.yaml")

    if os.path.exists(path):
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))

    return cfg

CONFIG = load_config("configs/local.yaml")