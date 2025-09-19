import os
import torch
import torch.nn as nn
import torch.distributed as dist

from PIL import Image

import hydra
from omegaconf import OmegaConf

import numpy as np

from modeling_slot_qformer import SlotQFormerModel

from utils.config import build_config


if __name__ == "__main__":
    cfg, cfg_yaml = build_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    os.makedirs(cfg.result_file_path, exist_ok=True)

    model = SlotQFormerModel.from_pretrained(
        "KU-AGILab/Slot_Q-Former",
    ).wrapper.to(device)
    model.freeze()
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

