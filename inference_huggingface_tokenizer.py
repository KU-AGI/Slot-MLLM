import os
import torch
import json
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput
from inference_tokenizer import SlotInferenceWrapper
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from torchvision import transforms
from PIL import Image

def get_transform(type='clip', keep_ratio=True, image_size=224, normalize=True):
    if type == 'clip':
        transform = []
        if keep_ratio:
            transform.extend([
                transforms.Resize(image_size, antialias=True),
                transforms.CenterCrop(image_size),
            ])
        else:
            transform.append(transforms.Resize((image_size, image_size), antialias=True))
        transform.extend([
            transforms.ToTensor(),
        ])
        if normalize:
            transform.append(
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            )

        return transforms.Compose(transform)
    else:
        raise NotImplementedError

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
        transform_config=None,
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
        self.transform_config = transform_config or {
            "type": "clip",
            "image_size": 448,
            "keep_ratio": False,
            "normalize": False
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
            "stage2": self.stage2,
            "transform_config": self.transform_config
        }

class SlotQFormerModel(PreTrainedModel):
    config_class = SlotQFormerConfig
    base_model_prefix = "slot_qformer"

    def __init__(self, config):
        super().__init__(config)
        # config를 OmegaConf로 변환
        cfg_dict = config.to_dict()
        cfg = OmegaConf.create(cfg_dict)
        self.wrapper = SlotInferenceWrapper(cfg)
        
        # 이미지 변환 설정
        self.transform = get_transform(
            type=config.transform_config["type"],
            image_size=config.transform_config["image_size"],
            keep_ratio=config.transform_config["keep_ratio"],
            normalize=config.transform_config["normalize"]
        )

    def process_image(self, image):
        """이미지를 처리합니다.
        
        Args:
            image: PIL.Image 또는 이미지 경로
            
        Returns:
            torch.Tensor: 처리된 이미지 텐서 [1, 3, H, W]
        """
        if isinstance(image, str):
            image = Image.open(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be PIL.Image or image path")
            
        image = self.transform(image).unsqueeze(0)  # [1, 3, H, W]
        return image

    def forward(self, pixel_values, return_dict=True):
        slot_tokens = self.wrapper.forward_stage_1(pixel_values)
        outputs = self.wrapper.forward_stage_2(slot_tokens)
        
        if return_dict:
            return BaseModelOutput(
                last_hidden_state=outputs["slots_1024"],
                hidden_states=None,
                attentions=None
            )
        return outputs["slots_1024"]

    def generate(self, image):
        """이미지를 생성합니다.
        
        Args:
            image: PIL.Image 또는 이미지 경로
            
        Returns:
            PIL.Image: 생성된 이미지
        """
        # 이미지 처리
        image_tensor = self.process_image(image)
        
        # 이미지 생성
        with torch.no_grad():
            slot_tokens = self.wrapper.forward_stage_1(image_tensor)
            outputs = self.wrapper.forward_stage_2(slot_tokens)
            generated_images = self.wrapper.generate_image(outputs)
        
        return generated_images[0]

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
                },
                transform_config={
                    "type": "clip",
                    "image_size": 448,
                    "keep_ratio": False,
                    "normalize": False
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