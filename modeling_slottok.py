import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.distributed as dist


import torchvision.transforms as transforms
from pytorch_lightning import LightningModule
import pyrootutils
from vector_quantize_pytorch import ResidualVQ

from einops import rearrange
from transformers import CLIPTokenizer

import numpy as np

from models.slottok.vit import Block
from models.slot_mllm_tokenizer import ImageTokenizer

from my_diffusers import DPMSolverMultistepScheduler

from my_diffusers import (
    DDPMScheduler,
    StableUnCLIPImg2ImgPipeline,
    UNet2DConditionModelWithAdapter,
)

from functools import partial

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

BOI_TOKEN = "<img>"
EOI_TOKEN = "</img>"
IMG_TOKEN = "<img_{:05d}>"

IMG_FLAG = "<image>"
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
IMAGE_ID_SHIFT = 32000

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


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


class SlotTrainingWrapper(LightningModule):
    """Training wrapper for Slot

    Args:
        LightningModule (cfg, model): model should be ImageTokenizer
    """

    def __init__(self, cfg, unCLIP_path="/path/to/unCLIP-SD"):
        super().__init__()
        self.cfg = cfg
        self.stage = cfg.experiment.stage

        # For guidance scale
        if hasattr(cfg.stage1, "guidance_scale") and cfg.stage1.guidance_scale is not None:
            self.guidance_scale = cfg.stage1.guidance_scale
        else:
            self.guidance_scale = 2.0       # Default guidance scale

        # Load tokenizer
        self.image_tokenizer = ImageTokenizer(
            model_path=cfg.checkpoint_path.model_path,
            diffusion_model_path=None,  # Diffusion model is loaded in TrainingWrapper
            device="cpu",  # For PyTorch Lightning
            load_diffusion=False,
            vq_type=cfg.stage2.vq.type,
            discarding_thre=cfg.stage2.vq.discarding_threshold,
            from_pretrained=True if cfg.checkpoint_path.model_path is not None else False,
            vit_precision=cfg.optimizer.vit_precision,
            diffusion_precision=cfg.optimizer.diffusion_precision,
            legacy=cfg.stage2.vq.legacy,
        )

        dinov2 = torch.hub.load("facebookresearch/dinov2", cfg.stage1.dino_model_name)
        self.backbone = DINOBackbone(dinov2)

        # Set backbone to half precision
        if cfg.optimizer.vit_precision == "fp16":
            self.backbone = self.backbone.half()

        visual_hidden_dim = self.backbone.dinov2.num_features

        if hasattr(cfg.stage1, "layer_norm") and cfg.stage1.layer_norm:
            print("Use LayerNorm for visual embedding")
            self.visual_embedding_layernorm = nn.LayerNorm(visual_hidden_dim)
        else:
            self.visual_embedding_layernorm = nn.Identity()

        if hasattr(cfg.stage1, "visual_embedding_encoder_as_mlp") and cfg.stage1.visual_embedding_encoder_as_mlp:
            print("Use MLP for visual embedding encoder")
            self.visual_embedding_encoder = nn.Sequential(
                nn.Linear(visual_hidden_dim, visual_hidden_dim),
                nn.ReLU(),
                nn.Linear(visual_hidden_dim, 1408)
            )
        else:
            self.visual_embedding_encoder = nn.Linear(visual_hidden_dim, 1408)

        self.image_tokenizer.model.visual_encoder = None

        self.out_layer_norm = nn.LayerNorm(768)
        self.out_linear_1024 = nn.Linear(768, 1024)

        ### For diffusion DDP
        diffusion_precision = "fp16" if cfg.optimizer.diffusion_precision else "fp32"
        pretrained_model_name = unCLIP_path
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

        # Added : slotadapt for unet
        if hasattr(self.cfg.stage1, "use_unet_slotadapt") and self.cfg.stage1.use_unet_slotadapt:
            self.use_unet_slotadapt = True

            unet_config = dict(self.unet.config)
            unet_with_slots = UNet2DConditionModelWithAdapter(**unet_config)

            state_dict = self.unet.state_dict()
            model_state_dict = unet_with_slots.state_dict()
            for k, v in state_dict.items():
                if k in model_state_dict:
                    model_state_dict[k] = v
                    
            unet_with_slots.load_state_dict(model_state_dict, strict=False)

            self.unet = unet_with_slots.to(dtype=torch.float32)
            self.diffusion_model.unet = self.unet
        else:
            self.use_unet_slotadapt = False

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

        if hasattr(cfg.stage1, "use_slot"):
            self.use_slot = cfg.stage1.use_slot
        else:
            self.use_slot = True
        self.slot_num = 32
        if hasattr(cfg.stage1, "slot_config"):
            self.slot_config = cfg.stage1.slot_config
        else:
            self.slot_config = None

        if hasattr(cfg.stage1, "use_causal"):
            self.use_causal = cfg.stage1.use_causal
        else:
            self.use_causal = True

        if hasattr(cfg.stage1, "use_blip_itc") and cfg.stage1.use_blip_itc:
            # This may increase gpu memory usage
            print("Use BLIP ITC")
            self.use_blip_itc = cfg.stage1.use_blip_itc
        else:
            print("Not use BLIP ITC")
            self.use_blip_itc = False

        # For logging
        self.stage = cfg.experiment.stage

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

        if hasattr(cfg.stage1, "use_proj") and cfg.stage1.use_proj:
            print("Use projection layer")
            self.vision_proj = self.image_tokenizer.model.vision_proj
            self.text_proj = self.image_tokenizer.model.text_proj
        else:
            print("Not use projection layer")
            self.vision_proj = nn.Identity()
            self.text_proj = nn.Identity()

        if hasattr(self.cfg.dataset, "text_max_length"):
            self.text_max_length = self.cfg.dataset.text_max_length
        else:
            self.text_max_length = 128

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


            if hasattr(self.cfg.stage2.vq, "shared_codebook") and self.cfg.stage2.vq.shared_codebook:
                use_shared_codebook = True
            else:
                use_shared_codebook = False
            self.quantize = ResidualVQ(
                dim=image_feats_size,
                num_quantizers=self.cfg.stage2.vq.num_quantizers,
                codebook_size=self.n_embed,
                codebook_dim=self.codebook_embed_dim,
                shared_codebook=use_shared_codebook,
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

    def setup(self, stage):
        # Freeze backbone
        if hasattr(self, "backbone") and self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Diffusion frozen
        if hasattr(self, "image_normalizer") and self.image_normalizer is not None:
            for param in self.image_normalizer.parameters():
                param.requires_grad = False
        if hasattr(self, "text_encoder") and self.text_encoder is not None:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        # Freeze VAE
        for param in self.vae.parameters():
            param.requires_grad = False
        # Freeze CLIP
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        for param in self.unet.parameters():
            param.requires_grad = False

        # casting to float32
        self.unet = self.unet.to(dtype=torch.float32)

        if self.stage == 1:
            if hasattr(self.cfg.stage1, "unfreeze_unet") and self.cfg.stage1.unfreeze_unet:
                if self.use_unet_slotadapt:
                    print("Unfreeze only slot adapter layers in UNet")
                    for name, param in self.unet.named_parameters():
                        if "adapter" in name:
                            param.requires_grad = True
                            print(f"Unfreeze (slot adapter): {name}")
                        else:
                            param.requires_grad = False
                if hasattr(self.cfg.stage1, "unfreeze_unet_crossattn") and self.cfg.stage1.unfreeze_unet_crossattn:
                    for name, param in self.unet.named_parameters():
                        if any(x in name for x in ["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out", "pos_emb"]):
                            print(f"Unfreeze {name}")
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

        # Freezing for stage 2
        if self.stage == 2:
            if hasattr(self.cfg.stage2, "unfreeze_unet") and self.cfg.stage2.unfreeze_unet:

                if self.use_unet_slotadapt:
                    print("Stage 2: unfreeze only slot adapter layers in UNet")
                    for name, param in self.unet.named_parameters():
                        if "adapter" in name:
                            param.requires_grad = True
                            print(f"Unfreeze (slot adapter): {name}")
                        else:
                            param.requires_grad = False
                else:
                    for name, param in self.unet.named_parameters():
                        if any(x in name for x in ["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out"]):
                            print(f"Unfreeze {name}")
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

            # Freeze stage 1 model
            for param in self.image_tokenizer.model.Qformer.parameters():
                param.requires_grad = False

            # Allow to train the out layer norm and out linear
            if hasattr(self.cfg.stage2, "unfreeze_linear") and \
                    self.cfg.stage2.unfreeze_linear:
                for param in self.out_linear_1024.parameters():
                    param.requires_grad = True
            else:
                for param in self.out_linear_1024.parameters():
                    param.requires_grad = False


    @torch.no_grad()
    def get_codebook_indices(self, img, flatten):
        batch = [img, "dummy_text"]
        image_feats = self.get_image_feats(batch, 0)["image_feats"]
        quant, embed_ind, _ = self.quantize(image_feats)

        if flatten:
            embed_ind = rearrange(embed_ind, "bs num_slots num_quantizers -> bs (num_slots num_quantizers)",
                                  num_quantizers=self.num_quantizers)  # embed_ind [bs, 128]

        return embed_ind

    @torch.no_grad()
    def get_codebook_entry(self, indices):
        ret = {}
        indices = rearrange(indices, "b (num_slots num_quantizers) -> b num_slots num_quantizers",
                            num_quantizers=self.num_quantizers)

        quant = self.quantize.get_output_from_indices(indices)  # quant [bs, 32, 768]

        reconstructed_image_feats = self.apply_transformer(quant, self.blocks,
                                                           self.pos_embed)  # reconstructed_image_feats [bs, 32, 768]

        reconstructed_image_feats_blocks_image_applied = self.apply_transformer(reconstructed_image_feats,
                                                                                self.blocks_image,
                                                                                self.pos_embed_image)  # reconstructed_image_feats_blocks_image_applied [bs, 32, 768]
        slots_1024 = self.convert_image_feats_to_slots(
            reconstructed_image_feats_blocks_image_applied)  # slots_1024 [bs, 32, 1024]
        ret["slots_1024"] = slots_1024

        # For class label input
        reverse_output_proj = self.get_mlp_decoded_embedding(
            reconstructed_image_feats_blocks_image_applied)  # reverse_output_proj [bs, 1024]
        ret["reverse_output_proj"] = reverse_output_proj

        return ret

    def encode_image(self, image_torch, flatten=True):
        if len(image_torch.shape) == 3:
            image_torch = image_torch.unsqueeze(0)

        img = image_torch
        with torch.no_grad():
            img_id = self.get_codebook_indices(img, flatten=flatten)

        return img_id

    def decode_image(self, indices, negative_indices=None, num_inference_steps=100):
        ret = self.get_codebook_entry(indices)

        self.diffusion_model.scheduler = self.val_schduler

        register = ret["slots_1024"].mean(dim=1, keepdim=True)  # [B, 1, D]
        image = self.diffusion_model(
            prompt_embeds=register,
            slot_embeds=ret["slots_1024"],
            height=self.image_size,
            width=self.image_size,
            guidance_scale=self.guidance_scale,
            num_inference_steps=num_inference_steps,
            image_embeds=ret["reverse_output_proj"],
        ).images

        return image

    def all_gather_with_grad(self, tensors):
        """
        Performs all_gather operation on the provided tensors.
        Graph remains connected for backward grad computation.
        """
        # Queue the gathered tensors
        world_size = torch.distributed.get_world_size()
        # There is no need for reduction in the single-proc case
        if world_size == 1:
            return tensors

        # tensor_all = GatherLayer.apply(tensors)
        tensor_all = GatherLayer.apply(tensors)

        return torch.cat(tensor_all, dim=0)

    def get_diffusion_noisy_model_input(self, batch):
        pixel_values = self.transform_256(batch[0])
        pixel_values = self.normalize_diffusion(pixel_values)

        # Convert images to latent space
        model_input = self.vae.encode(pixel_values).latent_dist.sample()
        model_input = model_input * self.vae.config.scaling_factor

        # Sample noise that we'll add to the model input
        noise = torch.randn_like(model_input)

        bsz, channels, height, width = model_input.shape
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
        )
        timesteps = timesteps.long()

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = self.scheduler.add_noise(
            model_input, noise, timesteps)

        return noisy_model_input, noise, timesteps, model_input

    def get_noisy_unclip_class_label(self, image_embeds):
        # Use same noise level for batch
        noise_level = torch.randint(
            0,
            self.image_noising_scheduler.config.num_train_timesteps,
            (1,),
            device=image_embeds.device
        ).item()

        noisy_image_embeds = self.diffusion_model.noise_image_embeddings(image_embeds, noise_level)
        return noisy_image_embeds

    def get_mlp_decoded_embedding(self, image_feats):
        reverse_output = self.image_down(image_feats)
        reverse_output = reverse_output.reshape(reverse_output.size(0), -1)
        reverse_output_proj = self.distill_image_proj(reverse_output)
        return reverse_output_proj

    @torch.no_grad()
    def get_clip_img_embedding(self, image):
        image = self.transform_224(image)
        image = self.normalize_clip(image)
        return self.image_encoder(image).image_embeds.to(self.device)

    def get_image_feats(self, batch, batch_idx: int, return_past_key_values=False, return_query_tokens=False):
        if len(batch) == 3:
            image, text, image_id = batch
        elif len(batch) == 2:
            image, text = batch
        else:
            raise ValueError(f"Unknown batch size {len(batch)}")

        # Normalize image
        image = self.normalize_vit(image)

        # DINO backbone은 FP32 연산을 기본으로 하므로,
        # 외부에서 autocast(FP16)가 걸려 있어도 여기서는 FP32로 강제한다.
        # (Input과 bias dtype mismatch 방지)
        with torch.no_grad():
            with autocast(enabled=False):
                image = image.to(dtype=torch.float32)
                image_embeds = self.backbone(image)  # [b, 1024, 16, 16]

        image_embeds = rearrange(image_embeds, "b d h w -> b (h w) d")  # [b, 256, 1024]
        image_embeds = self.visual_embedding_layernorm(image_embeds)

        image_embeds = self.visual_embedding_encoder(image_embeds)  # [b, 256, 1408]

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        query_tokens = self.image_tokenizer.model.query_tokens.expand(image_embeds.shape[0], -1, -1)

        # Assume image_embeds.shape[0] is the batch size (b) and you have 32 tokens (n)
        b, n, _ = query_tokens.shape

        query_output = self.image_tokenizer.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,  # We need to use cache for past_key_values
            return_dict=True,
            use_slot=self.use_slot,  # Important! Use slot
            slot_config=self.slot_config,
            use_causal=self.use_causal,
        )

        image_feats = query_output.last_hidden_state

        slot_mask = self.image_tokenizer.model.Qformer.bert.slot_guidance_mask

        if slot_mask is not None:
            slot_mask = torch.nan_to_num(slot_mask, nan=0.0, posinf=0.0, neginf=0.0)

        ret = {}
        ret["image_feats"] = image_feats
        ret["slot_mask"] = slot_mask

        if return_past_key_values:
            ret["past_key_values"] = query_output.past_key_values
        if return_query_tokens:
            ret["query_tokens"] = query_tokens

        return ret

    def get_text_feats(self, batch, batch_idx: int, return_text_tokens=False):
        if len(batch) == 3:
            image, text, image_id = batch
        elif len(batch) == 2:
            image, text = batch

        text_tokens = self.image_tokenizer.model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
        )

        text_output = self.image_tokenizer.model.Qformer.bert(
            text_tokens.input_ids.to(self.device),
            attention_mask=text_tokens.attention_mask.to(self.device),
            return_dict=True,
            is_decoder=True,
        )

        text_feat = text_output.last_hidden_state

        ret = {}
        ret["text_feats"] = text_feat

        if return_text_tokens:
            ret["text_tokens"] = text_tokens

        return ret

    def get_final_sep_indices(
            self,
            text,
    ):
        """
        Get list of text, then return the last [SEP] token index for each text.

        Args:
            text: List = [seq_len]

        Returns:
            final_sep_indices (List[int]):
        """
        final_sep_indices = []
        input_ids = self.image_tokenizer.model.tokenizer(
            text
        ).input_ids
        for input_id in input_ids:
            idx = min(len(input_id) - 1, self.text_max_length - 1)
            final_sep_indices.append(idx)

        return torch.tensor(final_sep_indices, dtype=torch.int)

    def get_itc_loss(self, batch, batch_idx: int,
                     is_validation=False,
                     image_feats=None,
                     past_key_values=None,
                     query_tokens=None,
                     image_embeds=None):
        logging_dict = {}

        if image_feats is None:
            image_feats = self.get_image_feats(batch, batch_idx)["image_feats"]

        b = image_feats.size(0)

        assert image_feats.size(-1) == 768

        image_feats_projected = self.vision_proj(image_feats)  # [b, 32, 768]
        image_feats_normalized = F.normalize(image_feats_projected, dim=-1)  # [b, 32, 256]
        if self.use_blip_itc:
            image_feats_last_token = image_feats_normalized
        else:
            image_feats_last_token = rearrange(image_feats_normalized[:, -1, :], "b d -> b 1 d").contiguous()

        text_feats_ret = self.get_text_feats(batch, batch_idx,
                                             return_text_tokens=True)  # [b, seq, 768]
        text_feats = text_feats_ret["text_feats"]
        text_tokens = text_feats_ret["text_tokens"]

        text_feats_projected = self.text_proj(text_feats)  # [b, seq, 256]
        text_feats_normalized = F.normalize(text_feats_projected, dim=-1)
        final_sep_indices = self.get_final_sep_indices(batch[1])
        text_feats_cls = text_feats_normalized[torch.arange(text_feats.size(0)), final_sep_indices]
        # text_feats_cls = text_feats_normalized[:, 0, :].contiguous()

        image_feats_last_token_gathered = self.all_gather_with_grad(image_feats_last_token)
        text_feats_cls_gathered = self.all_gather_with_grad(text_feats_cls)

        # Debug : Can I use Pytorch lightning all_gather?
        # image_feats_all = self.all_gather(image_feats, sync_grads=True)
        # text_feats_all = self.all_gather(text_feats, sync_grads=True)

        sim_q2t = torch.matmul(
            rearrange(image_feats_last_token, "bs n d -> bs 1 n d"),
            rearrange(text_feats_cls_gathered, "(bs ngpus) d -> (bs ngpus) d 1", bs=b)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        ###========= Last token =========###

        # Always use last token
        # sim_i2t = sim_q2t[:, :, -1]
        if self.use_blip_itc:
            # Use max token
            sim_i2t, _ = sim_q2t.contiguous().max(-1)
            sim_i2t = sim_i2t.contiguous()
        else:
            # Always use last token
            sim_i2t = sim_q2t
        # Debug : Test Original BLIP-2 loss
        # sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            rearrange(text_feats_cls, "bs d -> bs 1 1 d"),
            rearrange(image_feats_last_token_gathered, "(bs ngpus) n d -> (bs ngpus) d n", bs=b)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # Always use last token
        # sim_t2i = sim_t2q[:, :, -1]
        if self.use_blip_itc:
            # Use max token
            sim_t2i, _ = sim_t2q.contiguous().max(-1)
            sim_t2i = sim_t2i.contiguous()
        else:
            # Always use last token
            sim_t2i = sim_t2q
        # Debug : Test Original BLIP-2 loss
        # sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        rank = dist.get_rank()
        bs = image_feats.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            self.device
        )

        loss_itc = (
                           F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                           + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
                   ) / 2

        logging_dict["loss_itc"] = loss_itc

        ###========= Image Captioning =========###
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.image_tokenizer.model.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.image_tokenizer.model.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

        text_tokens.attention_mask = text_tokens.attention_mask.to(self.device)
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

        lm_output = self.image_tokenizer.model.Qformer(
            decoder_input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            past_key_values=past_key_values,
            return_dict=True,
            labels=labels.to(self.device),
            use_slot=self.use_slot,
            slot_config=self.slot_config,
        )

        loss_lm = lm_output.loss
        logging_dict["loss_lm"] = loss_lm

        cur_stage = "train" if not is_validation else "val"
        for key in logging_dict.keys():
            self.log(
                f"{cur_stage}/{key}",
                logging_dict[key],
                on_step=False if is_validation else True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        return logging_dict

    def convert_image_feats_to_slots(self, image_feats):
        slots = self.out_layer_norm(image_feats)
        slots_1024 = self.out_linear_1024(slots)

        return slots_1024

    def get_slot_embedding(self, batch, batch_idx: int, image_feats=None):
        if image_feats is None:
            image_feats = self.get_image_feats(batch, batch_idx)["image_feats"]
        image_feats_blocks_image_applied = self.apply_transformer(image_feats, self.blocks_image, self.pos_embed_image)
        slots_1024 = self.convert_image_feats_to_slots(image_feats_blocks_image_applied)
        return slots_1024

    def forward_stage_1(self, x):
        image_feats = self.get_image_feats((x, None), batch_idx=0)['image_feats']
        return image_feats

    def forward_stage_2(self, image_feats, quantize=False):
        ret = {}
        if quantize:
            if self.cfg.stage2.vq.vq_type == "mcq":
                quant, _, _ = self.quantize.forward_w_proj(image_feats)
            else:
                quant, _, _ = self.quantize(image_feats)
            reconstructed_image_feats = self.apply_transformer(quant, self.blocks, self.pos_embed)
        else:
            reconstructed_image_feats = image_feats

        reconstructed_image_feats_blocks_image_applied = self.apply_transformer(reconstructed_image_feats,
                                                                                self.blocks_image, self.pos_embed_image)
        slots_1024 = self.convert_image_feats_to_slots(reconstructed_image_feats_blocks_image_applied)
        ret["slots_1024"] = slots_1024

        # For class label input
        reverse_output_proj = self.get_mlp_decoded_embedding(reconstructed_image_feats_blocks_image_applied)
        ret["reverse_output_proj"] = reverse_output_proj

        return ret

    def generate_image(self, ret):
        self.diffusion_model.scheduler = self.val_schduler

        register = ret['slots_1024'].mean(dim=1, keepdim=True)  # [B, 1, D]
        return self.diffusion_model(
            prompt_embeds=register,
            slot_embeds=ret['slots_1024'],
            height=self.image_size,
            width=self.image_size,
            guidance_scale=self.guidance_scale,
            num_inference_steps=100,
            image_embeds=ret['reverse_output_proj'],
        ).images

    def get_stage1_loss(self, batch, batch_idx: int, is_validation=False):
        logging_dict = {}

        ret = self.get_image_feats(batch, batch_idx,
                                   return_past_key_values=True,
                                   return_query_tokens=True)
        image_feats = ret["image_feats"]
        past_key_values = ret["past_key_values"]
        query_tokens = ret["query_tokens"]
        # Slot attention mask
        slot_mask = ret.get("slot_mask", None)

        # itc loss calculation
        if self.use_itc:
            loss_itc = self.get_itc_loss(batch, batch_idx,
                                         is_validation, image_feats, past_key_values, query_tokens)

        # Bidirection decoder here

        # For diffusion input
        image_feats_blocks_image_applied = self.apply_transformer(image_feats, self.blocks_image, self.pos_embed_image)
        slots_1024 = self.convert_image_feats_to_slots(image_feats_blocks_image_applied)

        # For class label input
        reverse_output_proj = self.get_mlp_decoded_embedding(image_feats_blocks_image_applied)
        gt_img_clip_embed = self.get_clip_img_embedding(batch[0])
        loss_mse = F.mse_loss(reverse_output_proj, gt_img_clip_embed, reduction="mean")
        logging_dict["loss_mse"] = loss_mse

        # Diffusion loss calculation
        # Get the noisy model input
        noisy_model_input, noise, timesteps, model_input = self.get_diffusion_noisy_model_input(batch)
        class_labels = self.get_noisy_unclip_class_label(reverse_output_proj)

        # Predict the noise residual
        # Original
        if not self.use_unet_slotadapt:
            model_pred = self.unet(
                noisy_model_input, timesteps, slots_1024,
                class_labels=class_labels.detach(),
            ).sample
        else:
            register = slots_1024.mean(dim=1, keepdim=True)  # [B, 1, D]
            model_pred = self.unet(
                noisy_model_input,
                timesteps,
                encoder_hidden_states=register,
                slot_adapter_hidden_states=slots_1024,
                class_labels=class_labels.detach(),
            ).sample

        # Get the target for loss depending on the prediction type
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(
                model_input, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler.config.prediction_type}")

        # Compute instance loss
        loss_diffusion = F.mse_loss(model_pred.float(),
                                    target.float(), reduction="mean")
        logging_dict["loss_diffusion"] = loss_diffusion

        # Combine loss
        loss = 0.0
        if self.use_itc:
            loss += self.cfg.stage1.loss_weight.loss_itc * loss_itc["loss_itc"]
            loss += self.cfg.stage1.loss_weight.loss_lm * loss_itc["loss_lm"]
        loss += self.cfg.stage1.loss_weight.loss_diffusion * loss_diffusion
        loss += self.cfg.stage1.loss_weight.loss_mse * loss_mse

        # Use 3rd upsample block for alignment loss
        if self.use_unet_slotadapt:
            unet_adapter_attention_probs = self.unet.adapter_attention_probs

        if self.use_unet_slotadapt and \
            (slot_mask is not None) and \
                (unet_adapter_attention_probs is not None) and \
                    hasattr(self.cfg.stage1.loss_weight, "loss_slot_align") and \
                        self.cfg.stage1.loss_weight.loss_slot_align is not None and \
                            self.cfg.stage1.loss_weight.loss_slot_align > 0:
            ASA = rearrange(slot_mask, "b slots dim1024 -> b dim1024 slots").contiguous()
            b = ASA.shape[0]
            ADM_T = rearrange(unet_adapter_attention_probs, "(b heads) dim256 slots -> b heads dim256 slots", b=b).contiguous()
            ADM_T = ADM_T.mean(dim=1)

            ADM_T_up = F.interpolate(
                ADM_T.transpose(1, 2),
                size=ASA.shape[1],    # 1024 tokens
                mode="linear",
                align_corners=False
            ).transpose(1, 2)

            assert ADM_T_up.shape == ASA.shape

            eps = 1e-6
            ADM_prob = ADM_T_up.clamp(min=eps, max=1-eps)
            ADM_logits = torch.log(ADM_prob / (1 - ADM_prob))

            loss_slot_align = F.binary_cross_entropy_with_logits(
                ADM_logits,
                ASA,
            )

            logging_dict["loss_slot_align"] = loss_slot_align
            loss += self.cfg.stage1.loss_weight.loss_slot_align * loss_slot_align

        logging_dict["loss"] = loss

        # loss logging
        cur_stage = "train" if not is_validation else "val"
        for key in logging_dict.keys():
            self.log(
                f"{cur_stage}/{key}",
                logging_dict[key],
                on_step=False if is_validation else True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        return loss, image_feats, slots_1024, reverse_output_proj

    def calculate_rec_loss(self, rec, target):
        target = target / target.norm(dim=-1, keepdim=True)
        rec = rec / rec.norm(dim=-1, keepdim=True)
        rec_loss = (1 - (target * rec).sum(dim=-1)).mean()
        return rec_loss

    def apply_transformer(self, slots, transformer_blocks, pos_embed):
        pos_embed_applied_slot = slots + pos_embed.repeat(slots.size(0), 1, 1)
        # Apply Causal Transformer
        for blk in transformer_blocks:
            pos_embed_applied_slot = blk(pos_embed_applied_slot, use_causal_mask=False)
        return pos_embed_applied_slot

    def get_codebook_util(self, indices):
        num_codes = self.quantize.n_e
        uniques = indices.unique().numel()
        return (uniques / num_codes) * 100

    def get_stage2_loss(self, batch, batch_idx: int, is_validation=False):
        logging_dict = {}

        with torch.no_grad():
            ret = self.get_image_feats(batch, batch_idx,
                                   return_past_key_values=True,
                                   return_query_tokens=True)
            image_feats = ret["image_feats"]
            slot_mask = ret.get("slot_mask", None)

        # Quantize slots
        if self.cfg.stage2.vq.vq_type == "mcq":
            quant, embed_ind, loss_codebook = self.quantize.forward_w_proj(image_feats)
        else:
            quant, embed_ind, loss_codebook = self.quantize(image_feats)

        if self.cfg.stage2.vq.vq_type == "residual_vq" or self.cfg.stage2.vq.vq_type == "residual_sim_vq":
            loss_codebook = loss_codebook.sum()

        logging_dict["loss_codebook"] = loss_codebook

        # Flatten the quantized slots
        embed_ind = embed_ind.reshape(image_feats.shape[0], -1)

        # Reconstruct the slots
        reconstructed_image_feats = self.apply_transformer(quant, self.blocks, self.pos_embed)

        loss_recon = F.mse_loss(image_feats, reconstructed_image_feats, reduction="mean")
        logging_dict["loss_recon"] = loss_recon
        logging_dict["sim_768"] = F.cosine_similarity(image_feats, reconstructed_image_feats, dim=-1).mean()

        # Predict the noise residual
        # By using the reconstructed slot
        reconstructed_image_feats_blocks_image_applied = self.apply_transformer(
            reconstructed_image_feats, self.blocks_image, self.pos_embed_image
        )

        slots_1024 = self.convert_image_feats_to_slots(reconstructed_image_feats_blocks_image_applied)

        # For class label input
        reverse_output_proj = self.get_mlp_decoded_embedding(reconstructed_image_feats_blocks_image_applied)
        gt_img_clip_embed = self.get_clip_img_embedding(batch[0])
        loss_mse = F.mse_loss(reverse_output_proj, gt_img_clip_embed, reduction="mean")
        logging_dict["loss_mse"] = loss_mse

        # Compute diffusion loss
        noisy_model_input, noise, timesteps, model_input = self.get_diffusion_noisy_model_input(batch)
        class_labels = self.get_noisy_unclip_class_label(reverse_output_proj)

        if not self.use_unet_slotadapt:
            model_pred = self.unet(
                noisy_model_input, timesteps, slots_1024,
                class_labels=class_labels.detach(),
            ).sample
        else:
            register = slots_1024.mean(dim=1, keepdim=True)  # [B, 1, D]
            model_pred = self.unet(
                noisy_model_input,
                timesteps,
                encoder_hidden_states=register,
                slot_adapter_hidden_states=slots_1024,
                class_labels=class_labels.detach(),
            ).sample

        # Get the target for loss depending on the prediction type
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(
                model_input, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler.config.prediction_type}")

        # Compute instance loss
        loss_diffusion = F.mse_loss(model_pred.float(),
                                    target.float(), reduction="mean")
        logging_dict["loss_diffusion"] = loss_diffusion

        # Combine loss
        loss = self.cfg.stage2.loss_weight.loss_codebook * loss_codebook + \
               self.cfg.stage2.loss_weight.loss_recon * loss_recon + \
               self.cfg.stage2.loss_weight.loss_diffusion * loss_diffusion + \
               self.cfg.stage2.loss_weight.loss_mse * loss_mse

        unet_adapter_attention_probs = self.unet.adapter_attention_probs

        if self.use_unet_slotadapt and \
            (slot_mask is not None) and \
            (unet_adapter_attention_probs is not None) and \
            hasattr(self.cfg.stage2.loss_weight, "loss_slot_align") and \
            self.cfg.stage2.loss_weight.loss_slot_align is not None and \
            self.cfg.stage2.loss_weight.loss_slot_align > 0:
            ASA = rearrange(slot_mask, "b slots dim1024 -> b dim1024 slots").contiguous()
            b = ASA.shape[0]
            ADM_T = rearrange(unet_adapter_attention_probs, "(b heads) dim256 slots -> b heads dim256 slots", b=b).contiguous()
            ADM_T = ADM_T.mean(dim=1)

            ADM_T_up = F.interpolate(
                ADM_T.transpose(1, 2),
                size=ASA.shape[1],    # 1024 tokens
                mode="linear",
                align_corners=False
            ).transpose(1, 2)

            assert ADM_T_up.shape == ASA.shape

            eps = 1e-6
            ADM_prob = ADM_T_up.clamp(min=eps, max=1-eps)
            ADM_logits = torch.log(ADM_prob / (1 - ADM_prob))
            
            loss_slot_align = F.binary_cross_entropy_with_logits(
                ADM_logits,
                ASA,
            )

            logging_dict["loss_slot_align"] = loss_slot_align
            loss += self.cfg.stage2.loss_weight.loss_slot_align * loss_slot_align

        logging_dict["loss"] = loss

        # loss logging
        cur_stage = "train" if not is_validation else "val"
        for key in logging_dict.keys():
            self.log(
                f"{cur_stage}/{key}",
                logging_dict[key],
                on_step=False if is_validation else True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=False,
            )

        return loss, embed_ind, slots_1024, reverse_output_proj