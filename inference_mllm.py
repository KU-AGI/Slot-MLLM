import os
import argparse
from PIL import Image
import hydra
from omegaconf import OmegaConf
from einops import rearrange
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, seed_everything
from typing import List, Tuple, Optional

from modeling_slottok import SlotInferenceWrapper

class SlotMLLMInferenceWrapper(LightningModule):
    def __init__(self, model, visual_tokenizer, tokenizer, transform, special_tokens, is_14b=False, und_proj_path=None):
        super().__init__()
        self.model = model
        self.visual_tokenizer = visual_tokenizer
        self.text_tokenizer = tokenizer
        self.transform = transform
        self.is_14b = is_14b

        self.boi_token_id = special_tokens["boi_token"]
        self.eoi_token_id = special_tokens["eoi_token"]
        self.pad_token_id = special_tokens["pad_token"]
        self.text_vocab_size = special_tokens["text_vocab_size"]
        self.image_vocab_size = special_tokens["image_vocab_size"]
        self.image_token_length = 128
        self.last_image_token = self.image_vocab_size - 1

        self.boi_token = torch.tensor([self.boi_token_id], dtype=torch.long)
        self.eoi_token = torch.tensor([self.eoi_token_id], dtype=torch.long)
        self.pad_token = torch.tensor([self.pad_token_id], dtype=torch.long)

        self.generation_config = {
            "num_beams": 5,
            "max_new_tokens": 512
        }


        if hasattr(self.model, "config") and hasattr(self.model.config, "hidden_size"):
            d_model = self.model.config.hidden_size
        elif hasattr(self.model, "config") and hasattr(self.model.config, "n_embd"):
            d_model = self.model.config.n_embd
        else:
            d_model = self.model.get_input_embeddings().weight.shape[1]

        self.und_img_proj = nn.Sequential(
            nn.Linear(768, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        ).to(self.model.device, dtype=self.model.dtype)

        if und_proj_path and os.path.exists(und_proj_path):
            loaded_weights = torch.load(und_proj_path, map_location="cpu")
            self.und_img_proj.load_state_dict(loaded_weights)

    def _tokenize_segment(self, text, add_special_tokens=False):
        return self.text_tokenizer(text, add_special_tokens=add_special_tokens, return_tensors="pt").input_ids.squeeze(0)

    def _build_interleaved_prompt_continuous(self, prompt_str: str, img_feats_list: List[torch.Tensor]):
        segs = prompt_str.split("<img>")
        n_ph = len(segs) - 1
        if len(img_feats_list) != n_ph:
            raise ValueError(f"Mismatch: {n_ph} tags vs {len(img_feats_list)} images.")

        seg_ids = [self._tokenize_segment(seg, add_special_tokens=(i == 0)) for i, seg in enumerate(segs)]
        pieces = [seg_ids[0]]
        overwrites = []
        cur_len = seg_ids[0].numel()

        for k in range(n_ph):
            feat = img_feats_list[k]
            num_slots = feat.shape[0]
            placeholders = torch.full((num_slots,), self.pad_token_id, dtype=torch.long)
            img_block = torch.cat([self.boi_token, placeholders, self.eoi_token], dim=0)
            
            start_pos = cur_len + 1 
            overwrites.append((start_pos, feat))
            pieces.append(img_block)
            cur_len += img_block.numel()
            if seg_ids[k+1].numel() > 0:
                pieces.append(seg_ids[k+1])
                cur_len += seg_ids[k+1].numel()

        return torch.cat(pieces, dim=0), overwrites

    def _apply_continuous_overwrites(self, batch_input_ids, overwrites):
        embed_layer = self.model.get_input_embeddings()
        inputs_embeds = embed_layer(batch_input_ids).to(self.model.dtype)
        for (b_idx, start_pos, feat) in overwrites:
            proj = self.und_img_proj(feat.to(self.device, dtype=self.model.dtype))
            write_len = proj.shape[0]
            inputs_embeds[b_idx, start_pos : start_pos + write_len, :] = proj
        return inputs_embeds

    def _prepare_multimodal_input(self, prompt_str, image_path=None):
        img_feats_list = []
        if image_path:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                slot_embeds = self.visual_tokenizer.forward_stage_1(image) 
                img_feats_list = [slot_embeds[0]]

        input_ids, overwrites = self._build_interleaved_prompt_continuous(prompt_str, img_feats_list)
        input_ids = input_ids.unsqueeze(0).to(self.device)
        overwrites_batch = [(0, sp, feat) for sp, feat in overwrites]
        return self._apply_continuous_overwrites(input_ids, overwrites_batch)

    def visual_question_answering(self, prompt, image_path):
        if self.is_14b:
            prompt_str = self.text_tokenizer.apply_chat_template(
                [{"role": "user", "content": f"<img>\n{prompt}\nAnswer using a single word or phrase."}],
                tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_str = f"USER: <img>\n{prompt} Answer using a single word or phrase.\nASSISTANT:"
        
        inputs_embeds = self._prepare_multimodal_input(prompt_str, image_path)
        with torch.no_grad():
            output_ids = self.model.generate(inputs_embeds=inputs_embeds, max_new_tokens=50, num_beams=5)
        return self.text_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def captioning(self, input_image_path):
        prompt = "<img>Describe the image."
        if self.is_14b:
            prompt = self.text_tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"USER: {prompt}\nASSISTANT:"
        
        inputs_embeds = self._prepare_multimodal_input(prompt, input_image_path)
        with torch.no_grad():
            output_ids = self.model.generate(inputs_embeds=inputs_embeds, max_new_tokens=128, num_beams=5)
        return self.text_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def text_to_image_generation(self, prompt):
        prompt_str = f"USER: {prompt} Please generate an image.\nASSISTANT:"
        if self.is_14b:
            prompt_str = self.text_tokenizer.apply_chat_template([{"role": "user", "content": f"{prompt} Please generate an image."}], tokenize=False, add_generation_prompt=True)
        
        input_ids = self.text_tokenizer(prompt_str, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(input_ids=input_ids, max_new_tokens=512)
        return output_ids[:, input_ids.shape[1]:]

    def multimodal_prompt_image_generation(self, prompt, input_image_path):
        prompt_str = f"USER: <img>{prompt}\nASSISTANT:"
        if self.is_14b:
            prompt_str = self.text_tokenizer.apply_chat_template([{"role": "user", "content": f"<img>{prompt}"}], tokenize=False, add_generation_prompt=True)
        
        inputs_embeds = self._prepare_multimodal_input(prompt_str, input_image_path)
        with torch.no_grad():
            output_ids = self.model.generate(inputs_embeds=inputs_embeds, max_new_tokens=512)
        return output_ids

    def save_image(self, generate_id, save_path):
        try:
            idx_boi = (generate_id == self.boi_token_id).nonzero(as_tuple=True)[0]
            idx_eoi = (generate_id == self.eoi_token_id).nonzero(as_tuple=True)[0]
            
            if len(idx_boi) > 0 and len(idx_eoi) > 0:
                image_ids = generate_id[idx_boi[0]+1 : idx_eoi[0]]
            else:
                image_ids = generate_id[-(self.image_token_length):]

            image_ids = (image_ids - self.text_vocab_size).long()
            if image_ids.numel() < self.image_token_length:
                image_ids = torch.cat([image_ids, torch.zeros(self.image_token_length - image_ids.numel(), dtype=torch.long, device=image_ids.device)])
            else:
                image_ids = image_ids[:self.image_token_length]

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.float16):
                    img_output = self.visual_tokenizer.decode_image(image_ids.unsqueeze(0))
                
                img_output[0].save(save_path)
                print(f"Success: {save_path}")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # For tokenizer
    parser.add_argument("--slot_cfg_path", type=str, default="/path/to/config.yaml")
    parser.add_argument("--slot_ckpt_path", type=str, default="/path/to/slottok.ckpt")
    parser.add_argument("--unCLIP_path", type=str, default="/path/to/unCLIP-SD")
    
    # For mllm
    parser.add_argument("--mllm_model_path", type=str, default="/path/to/mllm_model")
    parser.add_argument("--und_proj_path", type=str, default="/path/to/und_proj.pth")

    # For inference
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="generated_images/")
    parser.add_argument("--generation", action="store_true", help="Whether to generate an image")
    parser.add_argument("--is_14b", action="store_true", help="Whether the model is 14B version")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set random seed
    seed_everything(42, workers=True)

    # Load tokenizer
    cfg = OmegaConf.load(args.slot_cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    visual_tokenizer = SlotInferenceWrapper.load_from_checkpoint(
        args.slot_ckpt_path,
        cfg=cfg,
        unCLIP_path=args.unCLIP_path,
        strict=False
    )

    if args.is_14b:
        text_tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-14B-Instruct",
        )
    else:
        text_tokenizer = AutoTokenizer.from_pretrained(
            "lmsys/vicuna-7b-v1.5",
        )
    
    # Load model 
    model = AutoModelForCausalLM.from_pretrained(
        args.mllm_model_path,
        torch_dtype=torch.bfloat16
    ).to(device)

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    # Set special tokens
    image_vocab_size = 8192
    text_vocab_size = text_tokenizer.vocab_size if not args.is_14b else model.config.vocab_size - 2 - image_vocab_size

    boi_token_id = text_vocab_size + image_vocab_size
    eoi_token_id = text_vocab_size + image_vocab_size + 1
    pad_token_id = text_tokenizer.pad_token_id if text_tokenizer.pad_token_id is not None else 0

    special_tokens = {
        "boi_token" : boi_token_id,
        "eoi_token" : eoi_token_id,
        "pad_token": pad_token_id,
        "text_vocab_size": text_vocab_size,
        "image_vocab_size": image_vocab_size,
    }
    print(f"Base LLM vocab size : {text_vocab_size}, Slot-MLLM vocab size: {model.config.vocab_size}")
    print(f"boi token id: {boi_token_id} | eoi token id: {eoi_token_id}")
    model = SlotMLLMInferenceWrapper(model, visual_tokenizer, text_tokenizer, transform, special_tokens, args.is_14b, args.und_proj_path).to(device)

    if args.generation:
        if args.prompt is None:
            raise ValueError("Please provide a prompt for image generation or editing.")
        
        if args.image_path is not None:
            ### Image Editing
            prompt = args.prompt
            input_image_path = args.image_path
            save_path = os.path.join(args.save_path, "edit_output_img.png")
            generated_ids = model.multimodal_prompt_image_generation(prompt, input_image_path)[0]
            model.save_image(generated_ids, save_path)
        else:
            ### Text-to-Image Generation
            prompt = args.prompt
            save_path = os.path.join(args.save_path, "t2i_img.png")
            generated_ids = model.text_to_image_generation(prompt)[0]
            model.save_image(generated_ids, save_path)
    else:
        if args.image_path is None:
            raise ValueError("Please provide an image path for image understanding.")
        
        if args.prompt is not None:
            ### Visual Question Answering
            prompt = args.prompt
            input_image_path = args.image_path
            response = model.visual_question_answering(prompt, input_image_path)
            print(response)
        else:
            ### Captioning
            input_image_path = args.image_path
            response = model.captioning(input_image_path)
            print(response)