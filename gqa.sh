CUDA_VISIBLE_DEVICES=0 python evaluation/gqa.py \
    --input=data/GQA/GQA_testdev_balanced_questions.json \
    --output=gqa.json \
    --vlm_module=models/llava-v1.6-vicuna-13b-hf \
    --planner=models/planner \
    --src=data/GQA/images \
    --model=reasoner \
    --grounding_basedir=tools/GroundingDINO