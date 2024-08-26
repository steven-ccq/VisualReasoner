CUDA_VISIBLE_DEVICES=0 python evaluation/textvqa.py \
    --input=data/TextVQA/TextVQA_0.5.1_val.json \
    --output=texvqa.json \
    --vlm_module=models/llava-v1.6-vicuna-13b-hf \
    --planner=models/planner \
    --src=data/TextVQA/test_images \
    --model=reasoner \
    --grounding_basedir=tools/GroundingDINO