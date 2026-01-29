import os
import torch
import torch.nn as nn
import torch.distributed as dist

from PIL import Image

import hydra
from omegaconf import OmegaConf
import argparse

from modeling_slottok import SlotTrainingWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="/path/to/config.yaml")
    parser.add_argument("--model_path", type=str, default="/path/to/slottok.ckpt")
    parser.add_argument("--unCLIP_path", type=str, default="/path/to/unCLIP-SD")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    os.makedirs(cfg.result_file_path, exist_ok=True)

    model = SlotTrainingWrapper.load_from_checkpoint(
        args.model_path,
        cfg=cfg,
        unCLIP_path=args.unCLIP_path,
        strict=False
    )
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

