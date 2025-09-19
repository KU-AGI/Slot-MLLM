import os
import torch
import json
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
import torchvision.transforms as transforms
import PIL
from einops import rearrange

import os
import torch
import torch.nn as nn
import torch.distributed as dist

from PIL import Image

import hydra
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, seed_everything
from omegaconf import OmegaConf
import pyrootutils

import torch.nn.functional as F
from pytorch_lightning.strategies import DDPStrategy
from einops import rearrange, einsum
import transformers

from functools import partial

import numpy as np

from models.slot_qformer.vit import Block
from models.slot_mllm_tokenizer import ImageTokenizer

from utils.config import build_config

from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableUnCLIPImg2ImgPipeline,
)

from vector_quantize_pytorch import ResidualVQ


class DINOBackbone(nn.Module):
    def __init__(self, dinov2):
        super().__init__()
        self.dinov2 = dinov2

    def forward(self, x):
        enc_out = self.dinov2.forward_features(x)
        return rearrange(
            enc_out["x_norm_patchtokens"],
            "b (h w ) c -> b c h w",
            h=int(np.sqrt(enc_out["x_norm_patchtokens"].shape[-2]))
        )

class SlotInferenceWrapper(LightningModule):
    """Inference wrapper for Slot

    Args:
        LightningModule (cfg, model): model should be ImageTokenizer
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.stage = 2

        # Load tokenizer
        self.image_tokenizer = ImageTokenizer(
            model_path=cfg.checkpoint_path.model_path,
            diffusion_model_path=None,  # Diffusion model is loaded in TrainingWrapper
            device="cpu",  # For PyTorch Lightning
            load_diffusion=False,
            vq_type=None,
            discarding_thre=None,
            from_pretrained=True if cfg.checkpoint_path.model_path is not None else False,
            vit_precision="fp16",
            diffusion_precision="fp16",
            legacy=False,
        )

        dinov2 = torch.hub.load("facebookresearch/dinov2", cfg.stage1.dino_model_name)
        self.backbone = DINOBackbone(dinov2)
        self.backbone = self.backbone.half()

        visual_hidden_dim = self.backbone.dinov2.num_features

        self.visual_embedding_layernorm = nn.LayerNorm(visual_hidden_dim)

        self.visual_embedding_encoder = nn.Sequential(
            nn.Linear(visual_hidden_dim, visual_hidden_dim),
            nn.ReLU(),
            nn.Linear(visual_hidden_dim, 1408)
        )

        self.image_tokenizer.model.visual_encoder = None

        self.out_layer_norm = nn.LayerNorm(768)
        self.out_linear_1024 = nn.Linear(768, 1024)

        ### For diffusion DDP
        diffusion_precision = "fp16"
        pretrained_model_name = "stabilityai/stable-diffusion-2-1-unclip"
        self.diffusion_model = StableUnCLIPImg2ImgPipeline.from_pretrained(pretrained_model_name,
                                                                           torch_dtype=
                                                                           dict(fp16=torch.float16, fp32=torch.float32)[
                                                                               diffusion_precision])

        self.feature_extractor = self.diffusion_model.feature_extractor
        self.image_encoder = self.diffusion_model.image_encoder
        self.image_normalizer = self.diffusion_model.image_normalizer
        self.image_noising_scheduler = self.diffusion_model.image_noising_scheduler
        if self.diffusion_model.text_encoder is not None:
            self.clip_tokenizer = self.diffusion_model.tokenizer
        else:
            self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        if self.diffusion_model.text_encoder is not None:
            self.text_encoder = self.diffusion_model.text_encoder
        self.unet = self.diffusion_model.unet
        self.vae = self.diffusion_model.vae
        self.unet = self.unet.to(dtype=torch.float32)

        ### For diffusion scheduler

        # Change to DDPMScheduler
        self.diffusion_model.scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name, subfolder="scheduler",
            torch_dtype=dict(fp16=torch.float16, fp32=torch.float32)[diffusion_precision])

        self.scheduler = self.diffusion_model.scheduler

        # Scheduler for validation
        scheduler_args = {}
        if "variance_type" in self.scheduler.config:
            variance_type = self.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        self.val_schduler = DPMSolverMultistepScheduler.from_config(
            self.scheduler.config, **scheduler_args
        )

        self.use_slot = True

        self.slot_num = 32
        self.slot_config = cfg.stage1.slot_config

        self.use_causal = True

        self.use_blip_itc = False

        # For logging

        self.image_size = cfg.stage1.image_size
        self.transform_256 = transforms.Resize((self.image_size, self.image_size), antialias=True)
        # Resize for CLIP input
        self.transform_224 = transforms.Resize((224, 224), antialias=True)

        self.normalize_diffusion = transforms.Normalize(mean=[0.5], std=[0.5])
        self.normalize_vit = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        # Normalize for CLIP input
        self.normalize_clip = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )

        self.save_path = None

        # Use unused model for stage 1, Quantize is not used
        self.image_tokenizer.model.encode_task_layer = None
        self.image_tokenizer.model.decode_task_layer = None
        self.image_tokenizer.model.quantize = None
        self.image_tokenizer.model.blocks = None
        self.image_tokenizer.model.blocks_image = None

        # itc loss
        self.use_itc = True
        self.temp = self.image_tokenizer.model.temp

        self.vision_proj = self.image_tokenizer.model.vision_proj
        self.text_proj = self.image_tokenizer.model.text_proj

        self.text_max_length = 32

        image_feats_size = 768

        self.pos_embed_image = nn.Parameter(torch.zeros(1, self.slot_num, image_feats_size))
        self.blocks_image = nn.ModuleList([
            Block(dim=image_feats_size,
                  num_heads=16,
                  mlp_ratio=4.0,
                  qkv_bias=True,
                  qk_scale=None,
                  drop=0.0,
                  attn_drop=0.0,
                  drop_path=0.0,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6)) for _ in range(self.cfg.stage2.blocks_layers)
        ])

        self.image_down = nn.Sequential(
            nn.Linear(image_feats_size, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 32, bias=False),
        )
        self.distill_image_proj = nn.Linear(self.slot_num * 32, 1024, bias=False)

        if self.stage == 2:
            self.codebook_embed_dim = self.cfg.stage2.vq.codebook_embed_dim
            self.n_embed = self.cfg.stage2.vq.n_embed  # 8192

            print(f"n_embed: {self.n_embed}, codebook_embed_dim: {self.codebook_embed_dim}")  # 32

            self.quantize = ResidualVQ(
                dim=image_feats_size,
                num_quantizers=self.cfg.stage2.vq.num_quantizers,
                codebook_size=self.n_embed,
                codebook_dim=self.codebook_embed_dim,
                shared_codebook=True,
            )
            self.num_quantizers = self.cfg.stage2.vq.num_quantizers

            self.pos_embed = nn.Parameter(torch.zeros(1, self.slot_num, image_feats_size))
            self.blocks = nn.ModuleList([
                Block(dim=image_feats_size,
                      num_heads=16,
                      mlp_ratio=4.0,
                      qkv_bias=True,
                      qk_scale=None,
                      drop=0.0,
                      attn_drop=0.0,
                      drop_path=0.0,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6)) for _ in range(self.cfg.stage2.blocks_layers)
            ])

            if self.cfg.stage2.use_blocks_image and \
                    self.cfg.stage2.blocks_image_layers is not None:

                self.use_blocks_image = True
                self.pos_embed_image = nn.Parameter(torch.zeros(1, self.slot_num, image_feats_size))
                self.blocks_image = nn.ModuleList([
                    Block(dim=768,
                          num_heads=16,
                          mlp_ratio=4.0,
                          qkv_bias=True,
                          qk_scale=None,
                          drop=0.0,
                          attn_drop=0.0,
                          drop_path=0.0,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6)) for _ in
                    range(self.cfg.stage2.blocks_image_layers)
                ])
            else:
                self.use_blocks_image = False

    def get_image_feats(self, batch, batch_idx: int):
        """Extract image features using the backbone and Q-former"""
        if len(batch) == 3:
            image, text, image_id = batch
        elif len(batch) == 2:
            image, text = batch
        else:
            raise ValueError(f"Unknown batch size {len(batch)}")

        # Normalize image
        image = self.normalize_vit(image)

        with torch.no_grad():
            image_embeds = self.backbone(image)  # [b, 1024, 16, 16]

        image_embeds = rearrange(image_embeds, "b d h w -> b (h w) d")  # [b, 256, 1024]
        image_embeds = self.visual_embedding_layernorm(image_embeds)
        image_embeds = self.visual_embedding_encoder(image_embeds)  # [b, 256, 1408]

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        query_tokens = self.image_tokenizer.model.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.image_tokenizer.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
            use_slot=True,
            slot_config=self.slot_config,
            use_causal=True,
        )

        return {"image_feats": query_output.last_hidden_state}

    def forward_stage_1(self, x):
        """First stage of the forward pass"""
        image_feats = self.get_image_feats((x, None), batch_idx=0)['image_feats']
        quant, embed_ind, _ = self.quantize(image_feats)
        return embed_ind

    def forward_stage_2(self, embed_ind):
        """Second stage of the forward pass"""
        ret = {}
        quant = self.quantize.get_output_from_indices(embed_ind)
        reconstructed_image_feats = self.apply_transformer(quant, self.blocks, self.pos_embed)

        reconstructed_image_feats_blocks_image_applied = self.apply_transformer(
            reconstructed_image_feats, 
            self.blocks_image, 
            self.pos_embed_image
        )
        slots_1024 = self.convert_image_feats_to_slots(reconstructed_image_feats_blocks_image_applied)
        ret["slots_1024"] = slots_1024

        # For class label input
        reverse_output_proj = self.get_mlp_decoded_embedding(reconstructed_image_feats_blocks_image_applied)
        ret["reverse_output_proj"] = reverse_output_proj

        return ret

    def decode_image(self, image_ids):
        ret = self.forward_stage_2(image_ids)
        return self.generate_image(ret)

    def generate_image(self, ret):
        """Generate image using the diffusion model"""
        self.diffusion_model.scheduler = self.val_schduler

        return self.diffusion_model(
            prompt_embeds=ret['slots_1024'],
            height=self.image_size,
            width=self.image_size,
            guidance_scale=2,
            num_inference_steps=100,
            image_embeds=ret['reverse_output_proj'],
        ).images

    def apply_transformer(self, slots, transformer_blocks, pos_embed):
        """Apply transformer blocks to the slots"""
        pos_embed_applied_slot = slots + pos_embed.repeat(slots.size(0), 1, 1)
        for blk in transformer_blocks:
            pos_embed_applied_slot = blk(pos_embed_applied_slot, use_causal_mask=False)
        return pos_embed_applied_slot

    def convert_image_feats_to_slots(self, image_feats):
        """Convert image features to slots"""
        slots = self.out_layer_norm(image_feats)
        slots_1024 = self.out_linear_1024(slots)
        return slots_1024

    def get_mlp_decoded_embedding(self, image_feats):
        """Get MLP decoded embedding from image features"""
        reverse_output = self.image_down(image_feats)
        reverse_output = reverse_output.reshape(reverse_output.size(0), -1)
        reverse_output_proj = self.distill_image_proj(reverse_output)
        return reverse_output_proj

class SlotQFormerConfig(PretrainedConfig):
    model_type = "slot_qformer"

    def __init__(
        self,
        image_size=256,
        slot_num=32,
        codebook_embed_dim=32,
        n_embed=8192,
        num_quantizers=4,
        blocks_layers=4,
        blocks_image_layers=4,
        use_blocks_image=True,
        bypass_codebook=False,
        use_causal=True,
        use_slot=True,
        slot_config=None,
        checkpoint_path=None,
        stage1=None,
        stage2=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.slot_num = slot_num
        self.codebook_embed_dim = codebook_embed_dim
        self.n_embed = n_embed
        self.num_quantizers = num_quantizers
        self.blocks_layers = blocks_layers
        self.blocks_image_layers = blocks_image_layers
        self.use_blocks_image = use_blocks_image
        self.bypass_codebook = bypass_codebook
        self.use_causal = use_causal
        self.use_slot = use_slot
        self.slot_config = slot_config or {
            "T": 1,
            "num_iterations": 3,
            "use_half_slot": False
        }
        self.checkpoint_path = checkpoint_path or {"model_path": None}
        self.stage1 = stage1 or {
            "dino_model_name": "dinov2_vitl14",
            "unfreeze_unet": True,
            "unfreeze_resnet": False,
            "image_size": 256,
            "loss_weight": {
                "loss_itc": 0.5,
                "loss_lm": 0.5,
                "loss_diffusion": 1,
                "loss_mse": 0.5
            },
            "use_causal": True,
            "use_slot": True,
            "slot_config": {
                "T": 1,
                "num_iterations": 3,
                "use_half_slot": False
            }
        }
        self.stage2 = stage2 or {
            "loss_weight": {
                "loss_codebook": 1,
                "loss_recon": 1,
                "loss_diffusion": 0.1,
                "loss_mse": 0.1
            },
            "unfreeze_unet": False,
            "unfreeze_linear": False,
            "blocks_layers": 4,
            "blocks_image_layers": 4,
            "use_blocks_image": True,
            "unclip": False,
            "vq": {
                "vq_type": "residual_vq",
                "num_quantizers": 4,
                "codebook_embed_dim": 32,
                "n_embed": 8192
            },
            "bypass_codebook": False
        }

    def to_dict(self):
        return {
            "image_size": self.image_size,
            "slot_num": self.slot_num,
            "codebook_embed_dim": self.codebook_embed_dim,
            "n_embed": self.n_embed,
            "num_quantizers": self.num_quantizers,
            "blocks_layers": self.blocks_layers,
            "blocks_image_layers": self.blocks_image_layers,
            "use_blocks_image": self.use_blocks_image,
            "bypass_codebook": self.bypass_codebook,
            "use_causal": self.use_causal,
            "use_slot": self.use_slot,
            "slot_config": self.slot_config,
            "checkpoint_path": self.checkpoint_path,
            "stage1": self.stage1,
            "stage2": self.stage2
        }

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2)

class SlotQFormerModel(PreTrainedModel):
    config_class = SlotQFormerConfig
    base_model_prefix = "slot_qformer"

    def __init__(self, config):
        super().__init__(config)
        # config를 OmegaConf로 변환
        cfg_dict = config.to_dict()
        cfg = OmegaConf.create(cfg_dict)
        self.wrapper = SlotInferenceWrapper(cfg)
        self.depth = 4
        image_size = 448
        self.image_process = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
        ])

    def forward(self, pixel_values, return_dict=True):
        with torch.autocast("cuda", dtype=torch.float16):
            # If pixel_values is PIL Image, convert to tensor
            if isinstance(pixel_values, PIL.Image.Image):
                pixel_values = self.image_process(pixel_values)
            if pixel_values.ndim == 3:
                pixel_values = pixel_values.unsqueeze(0)
            slot_tokens = self.wrapper.forward_stage_1(pixel_values)
            outputs = self.wrapper.forward_stage_2(slot_tokens)
            
            if return_dict:
                return BaseModelOutput(
                    last_hidden_state=outputs["slots_1024"],
                    hidden_states=None,
                    attentions=None
                )
            return outputs["slots_1024"]
    
    def forward_stage_1(self, pixel_values):
        with torch.autocast("cuda", dtype=torch.float16):
            if isinstance(pixel_values, PIL.Image.Image):
                pixel_values = self.image_process(pixel_values)
            if pixel_values.ndim == 3:
                pixel_values = pixel_values.unsqueeze(0)
            pixel_values = pixel_values.to(self.wrapper.device).half()
            slot_tokens = self.wrapper.forward_stage_1(pixel_values)
            return slot_tokens
    
    def forward_stage_2(self, slot_tokens):
        with torch.autocast("cuda", dtype=torch.float16):
            ret = self.wrapper.forward_stage_2(slot_tokens)
            return ret
        
    def encode_image(self, pixel_values):
        slot_tokens = self.forward_stage_1(pixel_values)
        slot_tokens = rearrange(slot_tokens, "b n_slots d -> b (n_slots d)", d=self.depth)
        return slot_tokens

    def decode_image(self, image_ids):
        if image_ids.ndim == 1:
            # batch size 1
            image_ids = image_ids.unsqueeze(0)

        if image_ids.ndim == 2:
            # Reshape to (batch_size, slots, depth)
            image_ids = rearrange(image_ids, 'b (slots depth) -> b slots depth', depth=self.depth)
            
        with torch.autocast("cuda", dtype=torch.float16):
            return self.wrapper.decode_image(image_ids)

    def generate_image(self, ret):
        with torch.autocast("cuda", dtype=torch.float16):
            return self.wrapper.generate_image(ret)

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        torch.save(self.wrapper.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from huggingface_hub import hf_hub_download
        
        # Hugging Face Hub 모델인 경우
        if "/" in pretrained_model_name_or_path and not os.path.exists(pretrained_model_name_or_path):
            # config.json 다운로드
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="config.json"
            )
            # pytorch_model.bin 다운로드
            model_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="pytorch_model.bin"
            )
            
            config = cls.config_class.from_dict(
                json.load(open(config_path))
            )
            model = cls(config)
            model.wrapper.load_state_dict(
                torch.load(model_path, map_location="cpu")
            )
            return model
            
        # 체크포인트 파일인 경우
        if pretrained_model_name_or_path.endswith('.ckpt') or pretrained_model_name_or_path.endswith('.pth'):
            # 기본 설정으로 config 생성
            config = cls.config_class(
                image_size=256,
                slot_num=32,
                codebook_embed_dim=32,
                n_embed=8192,
                num_quantizers=4,
                blocks_layers=4,
                blocks_image_layers=4,
                use_blocks_image=True,
                bypass_codebook=False,
                use_causal=True,
                use_slot=True,
                slot_config={
                    "T": 1,
                    "num_iterations": 3,
                    "use_half_slot": False
                }
            )
            model = cls(config)
            # 체크포인트에서 state_dict 로드
            checkpoint = torch.load(pretrained_model_name_or_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                model.wrapper.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.wrapper.load_state_dict(checkpoint, strict=False)
            return model
        
        # 로컬 디렉토리인 경우
        config = cls.config_class.from_dict(
            json.load(open(os.path.join(pretrained_model_name_or_path, "config.json")))
        )
        model = cls(config)
        model.wrapper.load_state_dict(
            torch.load(os.path.join(pretrained_model_name_or_path, "pytorch_model.bin"))
        )
        return model
    
    def to(self, device):
        self.wrapper.to(device)
        return self

def push_to_hub(
    model_path: str,
    repo_id: str,
    token: str,
    commit_message: str = "Add Slot MLLM model"
):
    """
    모델을 Hugging Face Hub에 업로드하는 함수
    
    Args:
        model_path (str): 모델이 저장된 경로
        repo_id (str): Hugging Face 저장소 ID (username/model-name 형식)
        token (str): Hugging Face API 토큰
        commit_message (str): 커밋 메시지
    """
    from huggingface_hub import HfApi, create_repo
    
    # 1. 모델 로드
    model = SlotQFormerModel.from_pretrained(model_path)
    
    # 2. 저장소 생성
    api = HfApi()
    try:
        create_repo(repo_id, token=token, repo_type="model")
    except Exception as e:
        print(f"Repository might already exist: {e}")
    
    # 3. 모델 업로드
    model.push_to_hub(
        repo_id,
        use_temp_dir=True,
        token=token,
        commit_message=commit_message
    )
    
    print(f"Model successfully pushed to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repository ID")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face API token")
    parser.add_argument("--commit_message", type=str, default="Add Slot MLLM model")
    
    args = parser.parse_args()
    push_to_hub(args.model_path, args.repo_id, args.token, args.commit_message) 