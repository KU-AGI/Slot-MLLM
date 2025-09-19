# Slot-MLLM: Object-Centric Visual Tokenization for Multimodal LLM

This repository contains the official implementation of the paper **Slot-MLLM: Object-Centric Visual Tokenization for Multimodal LLM**.

## Environment Setup

We provide a Conda configuration file to easily set up the environment:

```bash
conda env create -f slot_mllm.yaml
conda activate slot_mllm
```

## Download Model Weights

Please download the weights for each model component from the following links:

* **Slot Q-Former Weights:** [Download Slot Q-Former weights](https://drive.google.com/file/d/1ApDtlQwJnFizrIvYlElJg4y2ivbuuETW/view?usp=sharing)
* **Slot-MLLM Weights:** [Download Slot-MLLM weights](https://drive.google.com/drive/folders/1WPfkzejvJM_1Rpqs-31elZE7_sUdrsFz?usp=drive_link)

## Inference

### Slot Q-Former

Run the following command:

```bash
python inference_tokenizer.py
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
