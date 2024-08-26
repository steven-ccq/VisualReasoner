from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

from paddleocr import PaddleOCR
import numpy as np

import os
from groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert
import torch

from PIL import Image, ImageDraw, ImageFilter
import re
import json


class VLM_Module():
    def __init__(self, model_path, device='cuda:0'):
        self.device = device
        if 'llava' in model_path.lower():
            self.processor = LlavaNextProcessor.from_pretrained(model_path)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
            self.model_type = 'llava'
            self.prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{question} ASSISTANT:"
        elif 'blip2' in model_path.lower():
            self.processor = Blip2Processor.from_pretrained(model_path)
            self.model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
            self.model_type = 'blip2'
            self.prompt = "Question: {question} Answer:"
        elif 'instructblip' in model_path.lower():
            self.model = InstructBlipForConditionalGeneration.from_pretrained(model_path)
            self.processor = InstructBlipProcessor.from_pretrained(model_path)
            self.model_type = 'instructblip'
            self.prompt = "{question}"
        else:
            print('Undefined Model Type:')
            print(model_path)
            exit()
        self.model.to(device)
    def forward(self, image, question, max_new_tokens=16, do_sample=False):
        question = self.prompt.format(question=question)
        if self.model_type == 'llava':
            inputs = self.processor(question, image, return_tensors="pt").to(self.device)
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
            return self.processor.decode(output[0], skip_special_tokens=True).split('ASSISTANT:')[-1].strip()
        elif self.model_type == 'blip2':
            inputs = self.processor(image, question, return_tensors="pt").to(self.device, torch.float16)
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
            return self.processor.decode(output[0], skip_special_tokens=True).strip()
        elif self.model_type == 'instructblip':
            inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            return generated_text
    

class OCR_Module():
    def __init__(self):
        self.model = PaddleOCR(use_angle_cls=True, lang="en")
    def forward(self, img):
        np_img = np.array(img)
        result = self.model.ocr(np_img, cls=True)
        text = []
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                text.append({'label': line[1][0], 'score': line[1][1]})
        return text

    
class Grounding_Module():
    def __init__(self, base_dir):
        self.model = load_model(
            os.path.join(base_dir, "groundingdino/config/GroundingDINO_SwinT_OGC.py"), 
            os.path.join(base_dir, "weights/groundingdino_swint_ogc.pth")
            )

    def forward(self, img, prompt, bbox_thrd, text_thrd, do_clean=True):
        w, h = img.size
        img_source, img = load_image(image_path='', image=img)
        boxes, logits, phrases = predict(
            model=self.model,
            image=img,
            caption=prompt,
            box_threshold=bbox_thrd,
            text_threshold=text_thrd
            )
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        boxes = list(boxes)
        logits = logits.numpy()
        logits = list(logits)
        res = []
        for bbox, logit, phrase in zip(boxes, logits, phrases):
            res.append((list([int(xy) for xy in bbox]), logit, phrase))
        if do_clean:
            res = self._clean_bbox(res)
        return sorted(res, key=lambda x: x[1], reverse=True)
    
    def _clean_bbox(self, bbox_list):
        def get_range(bbox):
            return (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
        def check_recap(bbox1, bbox2):
            if bbox2[0]<bbox1[0] and bbox2[1]<bbox1[1] and bbox2[2]>bbox1[2] and bbox2[3]>bbox1[3]:
                return True
            return False

        bbox_list = sorted(bbox_list, key=lambda x: get_range(x[0]))
        cleaned_bbox_list = []
        for bbox in bbox_list:
            if len(bbox_list) == 0:
                cleaned_bbox_list.append(bbox)
                continue

            flag = True
            for cleaned_bbox in cleaned_bbox_list:
                if check_recap(cleaned_bbox[0], bbox[0]):
                    flag = False
                    break
            if flag:
                cleaned_bbox_list.append(bbox)
        return cleaned_bbox_list
    
def generate(model, processor, image, query, history):
    if type(image) == str:
        image = Image.open(image).convert('RGB')
    prompt = "USER: <image>\n{}\nASSISTANT:"
    input = 'Query:\n' + query + '\n' + 'Reasoning Path:\n' + str(history)
    prompt = prompt.format(input)
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, max_new_tokens=512)
    output = processor.decode(output[0], skip_special_tokens=True).split('ASSISTANT:')[-1].strip()
    try:
        output = output.replace("\'", "\"")
        output = json.loads(output)
    except:
        output = {'question': '', 'tool': '', 'operation': ''}
    return output

def highlight(image, bbox_list):
    np_img = np.array(image)
    np_ori = np_img.copy()
    np_img //= 4
    for bbox in bbox_list:
        np_img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = np_ori[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    image_h = Image.fromarray(np_img)
    return image_h

def check_blurry(sent):
    blurry_words = ["blurry", "hazy", "vague", "indistinct", "fuzzy", "unclear", "misty", "foggy", "muddled", "cloudy", "obscure", "out of focus", "blear", "blurred", "smeared", "shadowy", "veiled"]
    for word in blurry_words:
        if word in sent:
            return False
    return True

def grounding(grounding_module, image, ops, last_bbox, bbox_thrd=0.2, text_thrd=0.2):
    image_copy = image.copy()
    instruction = ops['instruction']
    bbox_list = re.findall(r'\[*?\]<BBOX>', instruction)
    if bbox_list and last_bbox:
        bbox = bbox_list[0]
        instruction = instruction.replace(bbox, 'red bounding box')
        draw = ImageDraw.Draw(image_copy)
        draw.rectangle(last_bbox[0], outline='red', width=3)
    target = ops['target']
    res = grounding_module.forward(image_copy, instruction, bbox_thrd, text_thrd, do_clean=True)
    res = [_ for _ in res if target in _[2]]
    return res

def answer(vlm_module, image, question):
    res = vlm_module.forward(image, question)
    return res

def ocr(ocr_module, image, last_bbox, thrd=0.9, max_num=3):
    image_copy = image.copy()
    w, h = image.size
    text = []
    if last_bbox:
        for bbox in last_bbox:
            cur_image = image_copy.crop(bbox)
            ratio = min(w//cur_image.size[0], h//cur_image.size[1])
            cur_image = cur_image.resize((cur_image.size[0]*ratio, cur_image.size[1]*ratio), Image.Resampling.LANCZOS)
            cur_image = cur_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
            cur_text = ocr_module.forward(cur_image)
            text += cur_text
    text = sorted(text, key=lambda x: x['score'], reverse=True)
    text = [ocr_tok['label'] for ocr_tok in text if ocr_tok['score']>=thrd][:max_num]
    res = ','.join(text)
    return res

def counting(grounding_module, image, ops, last_bbox):
    image_copy = image.copy()
    instruction = ops['instruction']
    bbox_list = re.findall(r'\[*?\]<BBOX>', instruction)
    if bbox_list and last_bbox:
        bbox = bbox_list[0]
        instruction = instruction.replace(bbox, 'red bounding box')
        draw = ImageDraw.Draw(image_copy)
        draw.rectangle(last_bbox[0], outline='red', width=3)
    target = ops['target']
    res = grounding_module.forward(image_copy, instruction, 0.2, 0.2, do_clean=False)
    res = [_[0] for _ in res if target in _[2]]
    return res


def execuate(module_dict, image, tool, question, ops, last_bbox=None):
    if tool == 'grounding':
        res = grounding(module_dict['grounding'], image, ops, last_bbox)
        if len(res) == 0:
            tool = 'answer'
        else:
            res = sorted(res, key=lambda x: x[1]) # (bbox, logit, phrase)
            return ('grounding', [bbox[0] for bbox in res])
    if tool == 'answer':
        res = answer(module_dict['answer'], image, question)
        return ('answer', res)
    if tool == 'ocr':
        res = ocr(module_dict['ocr'], image, last_bbox)
        return ('ocr', res)
    if tool == 'counting':
        res = counting(module_dict['grounding'], image, ops, last_bbox)
        return ('counting', res)

def call_planner(model, processor, image, query, history):
    response = generate(model, processor, image, query, history)
    return response

def reasoner(model, processor, module_dict, image, query, max_step=5):
    history = []
    last_bbox = None
    step = 1
    while step < max_step:
        # get current sub-question, tool and operations
        plan = call_planner(model, processor, image, query, history)
        question = plan['question']
        tool = plan['tool']
        ops = plan['operation']
        if tool == '':
            return execuate(module_dict, image, 'answer', question, ops, last_bbox)
        history.append(question)
        tool_used, res = execuate(module_dict, image, tool, question, ops, last_bbox, step_id=step)
        # execuate
        if tool_used != tool:
            tool_used, ans = execuate(module_dict, image, 'answer', query, '')
            return ans
        if tool_used == 'grounding':
            last_bbox = res
            step += 1
        else:
            if tool_used == 'ocr':
                text = res
                if text:
                    query = 'Here are words in this image: {}. Answer the following question with short answer: {}'.format(text, query)
                else:
                    query = 'Answer the following question with short answer: {}'.format(query)
                ans = execuate(module_dict, image, 'answer', query, '', None)[1]
                return ans
            elif tool_used == 'counting':
                bbox_list = res
                image = highlight(image, bbox_list)
                query = 'Answer the following question with a number: {}'.format(query)
                tool_used, ans = execuate(module_dict, image, 'answer', query, '')
                return ans
            elif tool_used == 'answer':
                tool_used, ans = execuate(module_dict, image, 'answer', query, '')
                return ans
            else:
                print('Undefined Tool', tool_used)
            break
    tool_used, ans = execuate(module_dict, image, 'answer', query, '')
    return ans