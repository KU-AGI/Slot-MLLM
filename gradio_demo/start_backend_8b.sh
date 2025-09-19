python3 gradio_demo/slot_mllm_flask.py \
    --image_transform configs/transform/slot_transform.yaml \
    --port 7890 \
    --llm_device cuda:0 \
    --tokenizer_device cuda:0 \
    --offload_encoder \
    --offload_decoder 
