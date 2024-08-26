CUDA_VISIBLE_DEVICES=0 python evaluation/stvqa.py \
    --input=data/ST-VQA/test_task_3.json \
    --output=stvqa.json \
    --vlm_module=models/llava-v1.6-vicuna-13b-hf \
    --planner=models/planner \
    --src=data/ST-VQA/images \
    --model=reasoner \
    --grounding_basedir=tools/GroundingDINO