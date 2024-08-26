CUDA_VISIBLE_DEVICES=0 python evaluation/tallyqa.py \
    --input=data/TallyQA/test.json \
    --output=tallyqa.json \
    --vlm_module=models/llava-v1.6-vicuna-13b-hf \
    --planner=models/planner \
    --src=data/TallyQA \
    --model=reasoner \
    --grounding_basedir=tools/GroundingDINO