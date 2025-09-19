python3 slot_mllm_flask.py \
    --image_transform configs/transform/slot_transform.yaml \
    --is_14b False \
    --port 7882 \
    --llm_device cuda:0 \
    --tokenizer_device cuda:0 \
