import os
from omegaconf import OmegaConf, DictConfig

# In this repo, use the CONFIG global variable for everything!

CONFIG = DictConfig({})


def load_config(filename, include_cmd_line=True):
    """ Loads configuration from default.yml, the config filename,
    and command line arguments, in that order. Returns nothing; it
    loads everything into the CONFIG global variable. """
    cfg = OmegaConf.load("configs/default.yaml")

    # platform-specific stuff for benchmarking
    platform_config = "configs/local.yaml"
    if os.path.exists(platform_config):
        cfg = OmegaConf.merge(cfg, OmegaConf.load(platform_config))

    if filename is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(filename))

    if include_cmd_line:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

    # try to come up with reasonable defaults
    if "cache_dir" not in cfg:
        cfg.cache_dir = "cache"

    for key in list(CONFIG.keys()):
        del CONFIG[key]

    CONFIG.update(cfg)


load_config(None)
