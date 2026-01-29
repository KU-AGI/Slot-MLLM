# Slot-MLLM: Object-Centric Visual Tokenization for Multimodal LLM

This repository contains the official implementation of the paper **Slot-MLLM: Object-Centric Visual Tokenization for Multimodal LLM**.

## Environment Setup

We provide a Conda configuration file to easily set up the environment:

```bash
conda env create -f slot_mllm.yaml
conda activate slot_mllm
```

## Huggingface Model Weights

* **unCLIP-SD Weights:** [unCLIP-SD](https://drive.google.com/drive/folders/1e27KOZZp0ZguivHZebo7GiKI61cWzRup?usp=drive_link)
* **SlotTok Weights:** [SlotTok](https://drive.google.com/drive/folders/1KuOdo41X41WwouAvMm1nm9OZVSNvJrYS?usp=drive_link)
* **Slot-MLLM Weights:** [Slot-MLLM-7B-instruct](https://drive.google.com/drive/folders/1l7UfGglplGSl5Ep8vZNK8xzkQk4SXLqh?usp=drive_link) | [Slot-MLLM-14B-instruct]()

## Inference

### SlotTok

Run the following command:

```bash
python inference_tokenizer.py --cfg_path=/path/to/config.yaml --model_path=/path/to/slottok.ckpt --unCLIP_path=/path/to/unCLIP-SD
```

### Slot-MLLM

Run the following command to perform each task:

```bash
# Image Captioning
python inference_mllm.py --image_path=sample_data/understanding_input_img.jpg --is_14b True/False
```

```bash
# Visual Question Answering
python inference_mllm.py --image_path=sample_data/understanding_input_img.jpg --prompt="What color is the small animal?" --is_14b True/False
```

```bash
# Text-to-Image Generation
python inference_mllm.py --prompt="A red bicycle against a blue wall." --generation --is_14b True/False
```

```bash
# Image Editing
python inference_mllm.py --image_path=sample_data/edit_input_img.png --prompt="leave only one cherry on top." --generation --is_14b True/False
```

## Guidelines for Responsible Use

Slot-MLLM is designed to effectively perform multimodal understanding and image generation tasks. To ensure responsible use, users are advised to adhere to the following:

* **Ethical Use:** Only utilize Slot-MLLM for ethical applications, clearly disclose generated content, and avoid biased or inappropriate data.
* **Validation:** Always validate and manually inspect generated outputs, particularly in sensitive or public-facing contexts.
* **Transparency:** Clearly communicate when outputs are AI-generated.
