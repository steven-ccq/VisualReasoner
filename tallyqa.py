import os
from utils import VLM_Module, OCR_Module, Grounding_Module
from PIL import Image, ImageDraw, ImageFilter
import json
import re
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
import tqdm
import argparse
import numpy as np
from utils import reasoner

vlm_module = None
ocr_module = None
grounding_module = None

processor = None
model = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--vlm_module')
    parser.add_argument('--planner')
    parser.add_argument('--src')
    parser.add_argument('--model', default='reasoner')
    parser.add_argument('--grounding_basedir')
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    output_file = output_file.format(args.vlm_module)
    src_dir = args.src
    vlm_module = VLM_Module(args.vlm_module)
    ocr_module = OCR_Module()
    grounding_module = Grounding_Module(args.grounding_basedir)
    processor = AutoProcessor.from_pretrained(args.planner)
    model = LlavaForConditionalGeneration.from_pretrained(args.planner, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to("cuda:0")

    module_dict = {
        'grounding': grounding_module,
        'answer': vlm_module,
        'ocr': ocr_module
    }

    with open(input_file, 'r') as f:
        eval_dataset = json.load(f)
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existed_ids = [json.loads(_)['question_id'] for _ in f]
        print('{} test existed'.format(len(existed_ids)))
        eval_dataset = [data for data in eval_dataset if data['question_id'] not in existed_ids]
    
    for data in tqdm.tqdm(eval_dataset):
        image_id = data['image']
        image_path = os.path.join(src_dir, image_id)
        image = Image.open(image_path).convert('RGB')
        question = data['question']
        question_id = data['question_id']
        answer = data['answer']
        is_simple = data['issimple']
        try:
            if args.model == 'reasoner':
                response = reasoner(model, processor, module_dict, image, question)
            elif args.model == 'raw':
                response = vlm_module.forward(image, question)
            with open(output_file, 'a') as f:
                f.write(json.dumps({'image': image_id, 'question': question, 'question_id': question_id, 'response': response, 'answer': answer, 'issimple': is_simple}))
                f.write('\n')
        except Exception as e:
            print(e)
            continue