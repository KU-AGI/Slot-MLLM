import hydra
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyrootutils
import os
import torch

from omegaconf import OmegaConf
from flask import Flask, request
import json
from typing import Optional
import transformers
from dataclasses import dataclass, field
import io
import base64
from PIL import Image
import gc
import logging
import traceback
import time
from modeling_slot_qformer import SlotQFormerModel
from inference_mllm import SlotMLLMInferenceWrapper

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

IMG_FLAG = '<image>'
NUM_IMG_TOKNES = 128
NUM_IMG_CODES = 8192

app = Flask(__name__)


def decode_image(encoded_image: str) -> Image:
    decoded_bytes = base64.b64decode(encoded_image.encode('utf-8'))
    buffer = io.BytesIO(decoded_bytes)
    image = Image.open(buffer)
    return image


def encode_image(image: Image.Image, format: str = 'PNG') -> str:
    with io.BytesIO() as buffer:
        image.save(buffer, format=format)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return encoded_image


@dataclass
class Arguments:
    image_transform: Optional[str] = field(default=None, metadata={"help": "config path of image transform"})
    is_14b: Optional[bool] = field(default=False, metadata={"help": "whether to use 14b model"})
    tokenizer: Optional[str] = field(default=None, metadata={"help": "config path of tokenizer used to initialize tokenizer"})
    model: Optional[str] = field(default=None, metadata={"help": "config path of llm"})
    port: Optional[str] = field(default=80, metadata={"help": "network port"})
    llm_device: Optional[str] = field(default='cuda:0', metadata={"help": "llm device"})
    tokenizer_device: Optional[str] = field(default='cuda:0', metadata={"help": "tokenizer device"})


parser = transformers.HfArgumentParser(Arguments)
args, = parser.parse_args_into_dataclasses()


class LLMService:
    def __init__(self, args) -> None:
        image_transform_cfg = OmegaConf.load(args.image_transform)

        self.image_transform = hydra.utils.instantiate(image_transform_cfg)
        self.image_tokenizer = SlotQFormerModel.from_pretrained("zheedong/Slot_Q-Former").to(args.tokenizer_device)

        model_name = "zheedong/Slot-MLLM" if not args.is_14b else "zheedong/Slot-MLLM-14B"

        self.text_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

        llm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        ).to(args.llm_device)

        # Set special tokens
        if args.is_14b:
            text_vocab_size = 152064    # TODO : hard code, need to be fixed
        else:
            text_vocab_size = self.text_tokenizer.vocab_size
        self.image_id_shift = text_vocab_size
        image_vocab_size = NUM_IMG_CODES
            
        self.model = llm_model
        self.llm_device = args.llm_device
        self.tokenizer_device = args.tokenizer_device
        self.boi_token_id = text_vocab_size + image_vocab_size
        self.eoi_token_id = text_vocab_size + image_vocab_size + 1
        self.tokenizer = self.text_tokenizer
        if args.is_14b:
            # self.tokenizer.bos_token = '<|endoftext|>'
            self.tokenizer.eos_token = '<|endoftext|>'

        print('Init Done...')


service = LLMService(args)


@app.route('/generate', methods=['GET', 'POST'])
def generate():
    debug = False
    if not debug:
        request_info = request.get_json()

        text_list = request_info['text'].split(IMG_FLAG)
        image_list = request_info['images']
        temperature = request_info.get('temperature', 0.7)
        num_beams = request_info.get('num_beams', 1)
        max_new_tokens = request_info.get('max_new_tokens', 256)
        top_p = request_info.get('top_p', 0.5)
        force_boi = request_info.get('force_boi', False)
    else:
        text_list = ['Please generate an image of a cat.']
        image_list = []
        temperature = 0.7
        num_beams = 1
        max_new_tokens = 256
        top_p = 0.5
        force_boi = False

    assert len(text_list) == len(image_list) + 1


    input_ids = None
    if len(image_list) > 0:
        images_tensor_list = []
        images_tensor_indices = []
        images_ids_list = []
        images_ids_indices = []
        for idx, image_item in enumerate(image_list):
            if isinstance(image_item, str):
                image = decode_image(image_item)
                image_tensor = service.image_transform(image)
                images_tensor_list.append(image_tensor)
                images_tensor_indices.append(idx)
            else:
                images_ids_list.append(image_item)
                images_ids_indices.append(idx)

        if len(images_tensor_list) > 0:
            images_tensor = torch.stack(images_tensor_list, dim=0).to(service.tokenizer_device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                images_ids_1 = service.image_tokenizer.encode_image(images_tensor).cpu()
            num_image_ids = images_ids_1.shape[-1]
        else:
            num_image_ids = len(images_ids_list[-1])
        images_ids_2 = torch.tensor(images_ids_list, dtype=torch.long)

        images_ids = torch.zeros((len(image_list), num_image_ids), dtype=torch.long)
        if len(images_tensor_indices) > 0:
            images_ids[images_tensor_indices, :] = images_ids_1
        if len(images_ids_indices) > 0:
            images_ids[images_ids_indices, :] = images_ids_2

        input_text = ''
        input_ids = []
        # Add BOS token to the first text_list item
        if service.tokenizer.bos_token is not None:
            text_list[0] = service.tokenizer.bos_token + text_list[0]
        else:
            text_list[0] = text_list[0]

        skip_idx = len(images_ids) - 3
        for i in range(images_ids.shape[0]):
            # We only use last two images
            if i in range(0, skip_idx + 1):
                input_text += text_list[i]
                text_ids = service.tokenizer(text_list[i], add_special_tokens=False, return_tensors='pt').input_ids
                input_ids.append(text_ids)
                continue

            single_image_ids = images_ids[i].view(-1).tolist()
            image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in single_image_ids]) + EOI_TOKEN
            input_text += text_list[i] + image_tokens

            # Get the input_ids of the image tokens
            image_ids = (torch.tensor(single_image_ids, dtype=torch.long) + service.image_id_shift).unsqueeze(0)
            image_ids = torch.cat(
                [torch.tensor([service.boi_token_id]).unsqueeze(0),
                 image_ids,
                 torch.tensor([service.eoi_token_id]).unsqueeze(0)], dim=1)
            text_ids = service.tokenizer(text_list[i], add_special_tokens=False, return_tensors='pt').input_ids
            input_ids.append(text_ids)
            input_ids.append(image_ids)


        input_text = input_text + text_list[-1]
        images_ids_list = images_ids.tolist()

        input_ids = torch.cat(input_ids, dim=1)
        input_ids = torch.cat(
            [input_ids, service.tokenizer(text_list[-1], add_special_tokens=False, return_tensors='pt').input_ids], dim=1)
    else:
        
        if service.tokenizer.bos_token is not None:
            input_text = service.text_tokenizer.bos_token + ''.join(text_list)
        else:
            input_text = ''.join(text_list)
        images_ids_list = []
        

    print(input_text)

    if input_ids is None:
        input_ids = service.tokenizer(input_text, add_special_tokens=False, return_tensors='pt').input_ids

    if force_boi:
        input_ids = torch.cat([input_ids, torch.tensor([service.boi_token_id]).unsqueeze(0)], dim=1)
    input_ids = input_ids.to(service.llm_device)

    generation_config = {
        'temperature': temperature,
        'num_beams': num_beams,
        'max_new_tokens': max_new_tokens,
        'top_p': top_p,
        'do_sample': True
    }

    generate_ids = service.model.generate(input_ids=input_ids, **generation_config)

    if force_boi:
        generate_ids = generate_ids[0][input_ids.shape[1] - 1:]
    else:
        generate_ids = generate_ids[0][input_ids.shape[1]:]
    print('generated_ids: ', generate_ids)
    boi_indices = torch.where(generate_ids == service.boi_token_id)[0].tolist()
    eoi_indices = torch.where(generate_ids == service.eoi_token_id)[0].tolist()
    # assert len(boi_indices) == len(eoi_indices)

    generated_image_base64_list = []
    text_mask = torch.ones_like(generate_ids, dtype=torch.bool)

    error_msg = []
    if len(boi_indices) != len(eoi_indices):
        error_msg.append(
            f'Num of BOI (begain of image) tokens: {len(boi_indices)} is not equal to EOI(end of image tokens): {len(eoi_indices)}, some image Some images will fail to decode.'
        )

    num_images = min(len(boi_indices), len(eoi_indices))
    for idx in range(num_images):
        boi_index, eoi_index = boi_indices[idx], eoi_indices[idx]
        # for boi_index, eoi_index in zip(boi_indices, eoi_indices):
        image_ids = generate_ids[boi_index + 1:eoi_index].unsqueeze(0).to(service.tokenizer_device)
        image_ids = image_ids - service.image_id_shift
        if image_ids.shape[-1] != NUM_IMG_TOKNES:
            error_msg.append(f'Len(image_ids) {image_ids.shape[-1]} is not equal to {NUM_IMG_TOKNES}')
            image_base64 = ''
        elif (image_ids < 0).any() or (image_ids >= NUM_IMG_CODES).any():
            error_msg.append(f'Some image_id out of range: [0, {NUM_IMG_CODES})')
            image_base64 = ''
        else:
            image = service.image_tokenizer.decode_image(image_ids)[0]
            image_base64 = encode_image(image)

        generated_image_base64_list.append(image_base64)
        text_mask[boi_index + 1:eoi_index] = False
        images_ids_list.append(image_ids.view(-1).tolist())
    generate_ids = generate_ids[text_mask]

    # print('generate_ids: ', generate_ids)
    # In Slot-MLLM, we don't have special tokens for image, so we need to skip them
    generate_text = service.tokenizer.decode(generate_ids, skip_special_tokens=True)
    # generate_text = service.tokenizer.decode(generate_ids, skip_special_tokens=False)
    # print('generate_text before: ', generate_text)
    if len(generate_ids) > 2 and generate_ids[-3] == service.boi_token_id and generate_ids[-2] == service.eoi_token_id:
        generate_text += IMG_FLAG
    generate_text = generate_text.replace(service.tokenizer.eos_token, '')
    print('generate_text: ', generate_text)
    return {'text': generate_text, 'images': generated_image_base64_list, 'images_ids': images_ids_list, 'error_msg': error_msg}


if __name__ == '__main__':
    # generate()
    app.run(host='0.0.0.0', port=args.port)
