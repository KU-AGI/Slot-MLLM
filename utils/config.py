import os
from datetime import datetime
from omegaconf import OmegaConf
from utils.utils import AttrDict

def override_from_file_name(cfg):
    c = OmegaConf.from_cli()
    if ('cfg_path' in c) and (not OmegaConf.is_missing(c, 'cfg_path')):
        c = OmegaConf.load(c.cfg_path)
    cfg = OmegaConf.merge(cfg, c)
    return cfg

def override_from_cli(cfg):
    c = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, c)
    return cfg

def to_attr_dict(cfg):
    c = OmegaConf.to_container(cfg)
    cfg = AttrDict.from_nested_dicts(c)
    return cfg

def build_config(path="configs/inference/slot_qformer_inference.yaml", struct=False):
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, struct)
    cfg = override_from_file_name(cfg)
    cfg = override_from_cli(cfg)

    if cfg.get('result_path', None) is not None:
        if cfg.eval or cfg.resume:
            # result_path = os.path.dirname(os.path.dirname(cfg.result_path))
            result_path = cfg.result_path
        else:
            now = datetime.now().strftime("%d%m%Y_%H%M%S")
            result_path = os.path.join(
                cfg.result_path, os.path.basename(cfg.cfg_path).split(".")[0], now
            )

        cfg.result_path = result_path

    cfg_yaml = OmegaConf.to_yaml(cfg)
    cfg = to_attr_dict(cfg) # TODO: using attr class in OmegaConf?

    return cfg, cfg_yaml
