import os
import torch
import json
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from inference_mllm import SlotMLLMInferenceWrapper
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
import torchvision.transforms as transforms
import PIL
from einops import rearrange
import hydra
from utils.config import build_config
from inference_tokenizer import SlotInferenceWrapper
from safetensors.torch import save_model

class SlotMLLMConfig(PretrainedConfig):
    model_type = "slot_mllm"

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

class SlotMLLMModel(PreTrainedModel):
    config_class = SlotMLLMConfig
    base_model_prefix = "slot_mllm"

    def __init__(self, config):
        super().__init__(config)
        # Convert config to OmegaConf
        cfg_dict = config.to_dict()
        cfg = OmegaConf.create(cfg_dict)
        
        # Initialize model components
        self.model = None  # Loaded from AutoModelForCausalLM
        self.visual_tokenizer = None  # Loaded from SlotInferenceWrapper
        self.text_tokenizer = None  # Loaded from AutoTokenizer
        self.transform = None  # Loaded from transform_cfg
        
        # Special token settings
        self.boi_token = None
        self.eoi_token = None
        self.text_vocab_size = None
        self.image_vocab_size = None
        self.image_token_length = 128
        self.last_image_token = None
        
        # Generation configuration
        self.generation_config = {
            "num_beams": 5,
            "max_new_tokens": 512
        }

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, return_dict=True):
        if pixel_values is not None:
            # Image encoding
            with torch.autocast("cuda", dtype=torch.float16):
                slot_tokens = self.visual_tokenizer.forward_stage_1(pixel_values)
                slot_tokens = slot_tokens + self.text_vocab_size
                slot_tokens = rearrange(slot_tokens, "b n d -> b (n d)", d=self.visual_tokenizer.num_quantizers)
                
                # Prepare inputs
                input_ids = self.prepare_input_ids(input_ids, slot_tokens)
        
        # Forward through the language model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        if return_dict:
            return CausalLMOutputWithPast(
                loss=outputs.loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions
            )
        return outputs

    def prepare_input_ids(self, prompt, img_ids):
        prompt_segs = prompt.split("<img>")
        prompt_seg_tokens = [
            self.text_tokenizer(seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids.squeeze(0)
            for i, seg in enumerate(prompt_segs)
        ]
        prompt_tokens = [prompt_seg_tokens[0]]
        for index in range(len(img_ids)):
            prompt_tokens.append(torch.cat([self.boi_token.to(self.device), img_ids[index], self.eoi_token.to(self.device)], dim=0))
            if prompt_seg_tokens[index + 1].shape[0] > 0:
                prompt_tokens.append(prompt_seg_tokens[index + 1])

        prompt_tokens = torch.cat(prompt_tokens, dim=0)
        return prompt_tokens.unsqueeze(0).to(self.device)

    def generate(self, input_ids=None, attention_mask=None, pixel_values=None, **kwargs):
        if pixel_values is not None:
            # Image encoding
            with torch.autocast("cuda", dtype=torch.float16):
                slot_tokens = self.visual_tokenizer.forward_stage_1(pixel_values)
                slot_tokens = slot_tokens + self.text_vocab_size
                slot_tokens = rearrange(slot_tokens, "b n d -> b (n d)", d=self.visual_tokenizer.num_quantizers)
                
                # Prepare inputs
                input_ids = self.prepare_input_ids(input_ids, slot_tokens)
        
        # Generation
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        # (1) config
        self.config.save_pretrained(save_directory)
        # (2) weights: use save_model which can handle shared buffers
        save_model(self, os.path.join(save_directory, "model.safetensors"))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, is_14b=False, **kwargs):
        # Load configuration
        config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        
        # Initialize model
        model = cls(config)
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load visual tokenizer
        visual_tokenizer_cfg_path = "configs/inference/slot_qformer_inference.yaml"
        visual_tokenizer_cfg, _ = build_config(path=visual_tokenizer_cfg_path)

        visual_tokenizer = SlotInferenceWrapper(visual_tokenizer_cfg).to(device)
        visual_tokenizer.load_state_dict(torch.load(visual_tokenizer_cfg.weight_path)["state_dict"], strict=False)
        visual_tokenizer.freeze()
        visual_tokenizer.eval()
        
        # Load text tokenizer
        if is_14b:
            text_tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-14B-Instruct"
            )
        else:
            text_tokenizer = AutoTokenizer.from_pretrained(
                "lmsys/vicuna-7b-v1.5",
            )
        
        # Load LLM model (using safetensors)
        llm_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        ).to(device)
        
        # Load transform configuration
        transform_cfg = OmegaConf.load(visual_tokenizer_cfg.transform_cfg_path)
        transform = hydra.utils.instantiate(transform_cfg)
        
        # Set special tokens
        text_vocab_size = text_tokenizer.vocab_size
        image_vocab_size = 8192
        
        boi_token_id = text_vocab_size + image_vocab_size
        eoi_token_id = text_vocab_size + image_vocab_size + 1
        special_tokens = {
            "boi_token": boi_token_id,
            "eoi_token": eoi_token_id,
            "text_vocab_size": text_vocab_size,
            "image_vocab_size": image_vocab_size,
        }

        # Add special tokens
        text_tokenizer.added_special_tokens = special_tokens
        text_tokenizer.boi_token = boi_token_id
        text_tokenizer.eoi_token = eoi_token_id
        
        # Set model components
        model.model = llm_model
        model.visual_tokenizer = visual_tokenizer
        model.text_tokenizer = text_tokenizer
        model.transform = transform
        model.boi_token = torch.tensor([special_tokens["boi_token"]], dtype=torch.int64)
        model.eoi_token = torch.tensor([special_tokens["eoi_token"]], dtype=torch.int64)
        model.text_vocab_size = special_tokens["text_vocab_size"]
        model.image_vocab_size = special_tokens["image_vocab_size"]
        model.image_token_length = 128
        model.last_image_token = model.image_vocab_size - 1

        if not is_14b:
            model.model.config.pad_token_id = text_tokenizer.pad_token_id
        else:
            model.model.config.pad_token_id = text_tokenizer.pad_token_id
        return model

def push_to_hub(
    model_path: str,
    repo_id: str,
    token: str,
    commit_message: str = "Add Slot MLLM model",
    is_14b: bool = False,
):
    """
    Upload a model to the Hugging Face Hub.
    
    Args:
        model_path (str): Path to the saved model directory.
        repo_id (str): Hugging Face repo ID (in the form username/model-name).
        token (str): Hugging Face API token.
        commit_message (str): Commit message.
    """
    from huggingface_hub import HfApi, create_repo, Repository
    import shutil
    import tempfile
    
    # 1. Load model
    model = SlotMLLMModel.from_pretrained(model_path, is_14b=is_14b)
    
    # 2. Create repository
    api = HfApi()
    try:
        create_repo(repo_id, token=token, repo_type="model")
    except Exception as e:
        print(f"Repository might already exist: {e}")
    
    # 3. Clone repository into a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Clone repository
        repo = Repository(
            local_dir=tmp_dir,
            clone_from=f"https://huggingface.co/{repo_id}",
            use_auth_token=token
        )
        
        # Configure LFS
        repo.lfs_track("*.safetensors")
        
        # Copy model files
        model.model.save_pretrained(
            tmp_dir,
            safe_serialization=True,
        )

        # model.model.config.save_pretrained(tmp_dir)
        model.text_tokenizer.save_pretrained(tmp_dir)

        # Commit and push changes
        repo.git_add()
        repo.git_commit(commit_message)
        repo.git_push()
    
    print(f"Model successfully pushed to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repository ID")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face API token")
    parser.add_argument("--commit_message", type=str, default="Add Slot MLLM model")
    parser.add_argument("--is_14b", type=bool, default=False, help="Use 14B model configuration")
    
    args = parser.parse_args()
    push_to_hub(args.model_path, args.repo_id, args.token, args.commit_message, args.is_14b)