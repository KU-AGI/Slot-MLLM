import os
import torch
import torch.nn as nn
import torch.distributed as dist

from PIL import Image

import hydra
from omegaconf import OmegaConf

from modeling_slottok import SlotTrainingWrapper

if __name__ == "__main__":
    cfg_path = "/path/to/config.yaml"
    cfg = OmegaConf.load(cfg_path)

    model_path = "/path/to/slottok.ckpt"
    unCLIP_path = "/path/to/unCLIP-SD"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    os.makedirs(cfg.result_file_path, exist_ok=True)

    model = SlotTrainingWrapper.load_from_checkpoint(model_path, cfg=cfg, strict=False)
    model = model.to(device)
    model.eval()

    image = Image.open("sample_data/sample_img.jpg")

    image = transform(image).unsqueeze(0).to(device)

    with torch.autocast("cuda", dtype=torch.float16):
        slot_tokens = model.forward_stage_1(image)
        print(slot_tokens)

        slots_1024 = model.forward_stage_2(slot_tokens)
        print(slots_1024)

        image = model.generate_image(slots_1024)
        image[0].save("sample_data/sample_img_reconstructed.jpg")

