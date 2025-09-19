import torch.nn as nn
import torch
import os
from typing import Any, Dict, List, Optional, Union
from transformers import (
    LlamaTokenizer,
    CLIPTokenizer,
)

# import torch
from PIL import Image
from torchvision import transforms

from .pipeline_stable_unclip_img2img import StableUnCLIPImg2ImgPipeline
from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DiffusionPipeline,
)

WEIGHTS_NAME = 'seed_quantizer.pt'
DIFFUSION_NAME = 'diffusion_model'


class ImageTokenizer(nn.Module):
    def __init__(self,
                 model_path,
                 diffusion_model_path=None,
                 load_diffusion=False,
                 image_size=224,
                 device='cuda',
                 from_pretrained=True,
                 vit_precision='fp16',
                 diffusion_precision='fp16',
                 unclip=True,
                 **kwargs):
        super().__init__()
        from .slot_qformer.qformer_quantizer import Blip2QformerQuantizer
        if not from_pretrained:
            print(f"$$ Warning: Loading model from scratch")
            model = Blip2QformerQuantizer(vit_precision=vit_precision, is_train=True, **kwargs)
        else:
            print(f"Loading model from {model_path}")
            # First, load the model from BLIP-2 Weights
            model = Blip2QformerQuantizer.from_pretrained(pretrained_model_path=model_path,
                                                        #vit_precision='fp16' if fp16 else 'fp32',
                                                        vit_precision=vit_precision,
                                                        is_train=True,
                                                        **kwargs)

        if diffusion_model_path is not None and load_diffusion:
            fp16 = True if diffusion_precision == 'fp16' else False
            # diffusion_model = DiffusionPipeline.from_pretrained(diffusion_model_path,
            #                                                     torch_dtype=torch.float16 if fp16 else torch.float32)
            
            if unclip:
                diffusion_model = StableUnCLIPImg2ImgPipeline.from_pretrained(diffusion_model_path,
                                                                            torch_dtype=dict(fp16=torch.float16, fp32=torch.float32)[diffusion_precision])
            # Test for Stable Diffusion 2.1
            else:
                pretrained_model_name = "stabilityai/stable-diffusion-2-1"
                diffusion_model = StableDiffusionPipeline.from_pretrained(pretrained_model_name,
                                                                        torch_dtype=dict(fp16=torch.float16, fp32=torch.float32)[diffusion_precision])
                diffusion_model.scheduler = DDPMScheduler.from_pretrained(
                    pretrained_model_name, subfolder="scheduler", torch_dtype=dict(fp16=torch.float16, fp32=torch.float32)[diffusion_precision])
                # vae = AutoencoderKL.from_pretrained(
                #     pretrained_model_name, subfolder="vae", torch_dtype=dict(fp16=torch.float16, fp32=torch.float32)[diffusion_precision])
                # unet = UNet2DConditionModel.from_pretrained(
                #     pretrained_model_name, subfolder="unet", torch_dtype=dict(fp16=torch.float16, fp32=torch.float32)[diffusion_precision])
                # diffusion_model = StableDiffusionPipeline(
                #     vae=vae,
                #     text_encoder=None,
                #     tokenizer=None,
                #     unet=unet,
                #     scheduler=scheduler,
                #     safety_checker=None,
                #     feature_extractor=None,
                #     requires_safety_checker=False,
                # )    
        else:
            diffusion_model = None
            self.diffusion_model = None

        model = model.to(device)
        if diffusion_model is not None:
            diffusion_model = diffusion_model.to(device)

        processor = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=3),
            # transforms.Resize(image_size, interpolation=3),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        shape_latents = torch.Size([1, 4, 96, 96])
        self.latents = torch.randn(shape_latents, generator=None, device=device, dtype=torch.bfloat16, layout=torch.strided)

        shape_noise = torch.Size([1, 1024])
        self.noise = torch.randn(shape_noise, generator=None, device=device, dtype=torch.bfloat16, layout=torch.strided)

        self.model = model
        if diffusion_model is not None:
            self.diffusion_model = diffusion_model
        self.processor = processor
        self.device = device
        self.fp16 = vit_precision == 'fp16'

    def __len__(self):
        return self.model.n_embed

    def encode(self, image_torch):
        '''Convert a batch of img to code
        Args:
            model: The tokenizer model.
            img: [b, c, h, w]
        '''
        if len(image_torch.shape) == 3:
            image_torch = image_torch.unsqueeze(0)

        # img = image_torch.to(self.device)
        img = image_torch
        if self.fp16:
            img = img.half()
        with torch.no_grad():
            # id is index of codebook
            id, _ = self.model.get_codebook_indices(img)
        return id.view(img.shape[0], -1)

    def decode(self, indices, negative_indices=None, guidance_scale=10, num_inference_steps=20):
        # image_embed = [1, 1024]
        image_embeds = self.model.get_codebook_entry(indices)
        # image = self.diffusion_model(image_embeds=image_embed,
        #                              noise_level=0,
        #                              num_inference_steps=20,
        #                              latents=self.latents,
        #                              noise=self.noise).images
        if negative_indices is not None:
            assert indices.shape == negative_indices.shape, 'Negative indices must have the same shape with indices'
            negative_image_embeds = self.model.get_codebook_entry(negative_indices)
        else:
            negative_image_embeds = None

        image = self.diffusion_model(
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            guidance_scale=guidance_scale,
            noise_level=0,
            num_inference_steps=num_inference_steps,
            latents=self.latents,
        ).images
        return image