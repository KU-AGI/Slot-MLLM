import os
import argparse
from PIL import Image
import hydra
from omegaconf import OmegaConf
from einops import rearrange
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pytorch_lightning import LightningModule, seed_everything

from utils.config import build_config
from modeling_slot_qformer import SlotQFormerModel

class SlotMLLMInferenceWrapper(LightningModule):
    def __init__(self, model, visual_tokenizer, tokenizer, transform, special_tokens, is_14b=False):
        super().__init__()
        self.model = model
        self.visual_tokenizer = visual_tokenizer
        self.text_tokenizer = tokenizer
        self.transform = transform
        self.is_14b = is_14b

        self.boi_token = torch.tensor([special_tokens["boi_token"]], dtype=torch.int64)
        self.eoi_token = torch.tensor([special_tokens["eoi_token"]], dtype=torch.int64)
        self.text_vocab_size = special_tokens["text_vocab_size"]
        self.image_vocab_size = special_tokens["image_vocab_size"]
        self.image_token_length = 128
        self.last_image_token = self.image_vocab_size - 1

        self.generation_config = {
            "num_beams": 5,
            "max_new_tokens": 512
        }

    def visual_question_answering(self, prompt, input_image_path):
        # Encode image
        image = Image.open(input_image_path)
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16):
            slot_tokens = self.visual_tokenizer.forward_stage_1(image)
        slot_tokens = slot_tokens + self.text_vocab_size
        slot_tokens = rearrange(slot_tokens, "b n d -> b (n d)", d=visual_tokenizer.num_quantizers)
        
        if self.is_14b:
            prompt = self.text_tokenizer.apply_chat_template([{"role" : "user", "content" : f"<img>{prompt} Please provide an accurate answer consisting of only one word or phrase."}], tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"USER: <img>{prompt} Please provide an accurate answer consisting of only one word or phrase.\nASSISTANT:"
        input_ids = self.prepare_input_ids(prompt, slot_tokens)

        with torch.no_grad():
            generate_ids = self.model.generate(
                input_ids=input_ids,
                **self.generation_config
            )
        generate_ids = generate_ids[0, input_ids.shape[1]:]
        response = self.text_tokenizer.decode(generate_ids, skip_special_tokens=True)
        print(response)
        return response


    def captioning(self, input_image_path):
        # Encode image
        image = Image.open(input_image_path)
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16):
            slot_tokens = self.visual_tokenizer.forward_stage_1(image)
        slot_tokens = slot_tokens + self.text_vocab_size
        slot_tokens = rearrange(slot_tokens, "b n d -> b (n d)", d=visual_tokenizer.num_quantizers)
        
        if self.is_14b:
            prompt = self.text_tokenizer.apply_chat_template([{"role" : "user", "content" : "<img> Please provide an accurate and concise description of the given image."}], tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"USER: <img> Please provide an accurate and concise description of the given image.\nASSISTANT:"
        input_ids = self.prepare_input_ids(prompt, slot_tokens)

        with torch.no_grad():
            generate_ids = self.model.generate(
                input_ids=input_ids,
                **self.generation_config
            )
        generate_ids = generate_ids[0, input_ids.shape[1]:]
        response = self.text_tokenizer.decode(generate_ids, skip_special_tokens=True)
        print(response)
        return response


    def text_to_image_generation(self, prompt):
        if self.is_14b:
            prompt = self.text_tokenizer.apply_chat_template([{"role" : "user", "content" : f"{prompt} Please generate an image."}], tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"USER: {prompt} Please generate an image.\nASSISTANT:"
        input_ids = self.text_tokenizer(prompt, add_special_tokens=True, return_tensors='pt').input_ids.to(self.device)
        
        with torch.no_grad():
            generate_ids = self.model.generate(
                input_ids=input_ids,
                **self.generation_config
            )
        generate_ids = generate_ids[:, input_ids.shape[1]:]
        return generate_ids
    

    def multimodal_prompt_image_generation(self, prompt, input_image_path):
        # Encode image
        image = Image.open(input_image_path)
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16):
            slot_tokens = self.visual_tokenizer.forward_stage_1(image)
        slot_tokens = slot_tokens + self.text_vocab_size
        slot_tokens = rearrange(slot_tokens, "b n d -> b (n d)", d=visual_tokenizer.num_quantizers)

        if self.is_14b:
            prompt = self.text_tokenizer.apply_chat_template([{"role" : "user", "content" : f"<img>{prompt}"}], tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"USER: <img>{prompt}\nASSISTANT:"
        input_ids = self.prepare_input_ids(prompt, slot_tokens)

        with torch.no_grad():
            generate_ids = self.model.generate(
                input_ids=input_ids,
                **self.generation_config
            )
        generate_ids = generate_ids[:, input_ids.shape[1]:]
        return generate_ids


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
        

    def save_image(self, generate_id, save_path):
        boi_list = torch.where(generate_id == self.boi_token.to(self.device))[0]
        eoi_list = torch.where(generate_id == self.eoi_token.to(self.device))[0]

        if len(boi_list) == 0 and len(eoi_list) == 0:
            return 
        
        elif len(boi_list) == 0 and len(eoi_list) != 0:
            eoi_index = eoi_list[0]
            image_ids = (generate_id[:eoi_index] - self.text_vocab_size)
        
        elif len(boi_list) != 0 and len(eoi_list) != 0:
            boi_index = boi_list[0]
            eoi_index = eoi_list[0]
            image_ids = (generate_id[boi_index+1:eoi_index] - self.text_vocab_size)

        else:
            return

        # Fill zeros
        if image_ids.shape[0] < self.image_token_length:
            image_ids = torch.cat([image_ids, torch.zeros(self.image_token_length - image_ids.shape[0], dtype=torch.int64).to(image_ids)], dim=0)
        else:
            image_ids = image_ids[:self.image_token_length]

        # Check token range
        if any(token < 0 or token > self.last_image_token for token in image_ids):
            print("Invalid token range")
            return
        
        # Decode image
        try:
            image_ids = rearrange(image_ids, "(n d) -> n d", d=visual_tokenizer.num_quantizers).unsqueeze(0)
            slots_1024 = self.visual_tokenizer.forward_stage_2(
                image_ids,
            )
            with torch.autocast("cuda", dtype=torch.float16):
                image = self.visual_tokenizer.generate_image(slots_1024)
            image[0].save(save_path)
            
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    visual_tokenizer_cfg_path = "configs/inference/slot_qformer_inference.yaml"
    visual_tokenizer_cfg, _ = build_config(path=visual_tokenizer_cfg_path)

    visual_tokenizer = SlotQFormerModel.from_pretrained(
        "KU-AGI/Slot_Q-Former",
    ).wrapper.to(device)
    visual_tokenizer.freeze()
    visual_tokenizer.eval()

    model_name = "KU-AGI/Slot-MLLM-7B-instruct" if not args.is_14b else "KU-AGI/Slot-MLLM-14B-instruct"

    text_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )

    # Load model 
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
    ).to(device)

    transform_cfg = OmegaConf.load(visual_tokenizer_cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    # Set special tokens
    image_vocab_size = 8192
    text_vocab_size = text_tokenizer.vocab_size if not args.is_14b else model.config.vocab_size - 2 - image_vocab_size

    boi_token_id = text_vocab_size + image_vocab_size
    eoi_token_id = text_vocab_size + image_vocab_size + 1
    special_tokens = {
        "boi_token" : boi_token_id,
        "eoi_token" : eoi_token_id,
        "text_vocab_size": text_vocab_size,
        "image_vocab_size": image_vocab_size,
    }
    print(f"Base LLM vocab size : {text_vocab_size}, Slot-MLLM vocab size: {model.config.vocab_size}")
    print(f"boi token id: {boi_token_id} | eoi token id: {eoi_token_id}")
    model = SlotMLLMInferenceWrapper(model, visual_tokenizer, text_tokenizer, transform, special_tokens, args.is_14b).to(device)

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
        else:
            ### Captioning
            input_image_path = args.image_path
            response = model.captioning(input_image_path)