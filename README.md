# VisualReasoner

Official repository for the paper "From the Least to the Most: Building a Plug-and-Play Visual Reasoner via Data Synthesis"

---

## Setup

```bash
git clone https://github.com/steven-ccq/VisualReasoner.git
cd VisualReasoner
```

### Environment

```bash
# Python 3.8
pip install -r requirements.txt
```

### Grounding DINO

```bash
cd tools
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e .
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

### Planner Model

Download the [adapter](https://drive.google.com/file/d/1NJpQuvMr2JS56j2qCZFOJ2L9QKeqOc8A/view?usp=sharing).

Merge it with [llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) to obtain the Planner model.

## Inference

First, download the corresponding test sets as guided in the `data/` directory.

To facilitate usage, we have provided scripts for each test task:

```bash
# TextVQA
bash textvqa.sh
# TallyQA
bash tallyqa.sh
# ST-VQA
bash stvqa.sh
# GQA
bash gqa.sh
```

The parameters used in the scripts are described in the table below:

| Argument        | Description                         |
|-----------------|-------------------------------------|
| `input`         | Path to the input file              |
| `output`        | Path to the output file             |
| `vlm_module`    | Path to the Answer model            |
| `src`           | Path to the image folder            |
| `model`         | Path to the Planner model           |
| `grounding_basedir` | Path to the Grounding DINO    |

---

## Evaluation

```bash
# TextVQA
python eval/eval_textvqa.py --input=textvqa.json
# TallyQA
python eval/eval_tallyqa.py --input=tallyqa.json
# ST-VQA
https://rrc.cvc.uab.es/?ch=11
# GQA
python eval/eval_gqa.py --input=gqa.json
```

## Data

We also provide a 1M dataset synthesized using the least-to-most method, which includes both [profile](https://drive.google.com/file/d/1ayPyNUWkkkC1BzATlNKQm_l-ihlKMICe/view?usp=sharing) and [dataset](https://drive.google.com/file/d/14m_LxYkMChLLQm7bTYS3hR6YjmdXCyuX/view?usp=sharing) files.

### profile
The profile contains information for each image, formatted as follows:

```json
{
    "image_id": "image id",
    "url": "URL to download the image",
    "profile": {
        "objs": [
            {"bbox": "bounding box (xyxy)", "label": "label of objects", "score": "confidence score"},
            ...
        ],
        "description": "detailed description of this image",
        "groups": [
            {"bbox": "bounding box (xyxy)", "obj_ids": "indexes of objects in the group, used to reference objects", "caption": "brief description of the group content"},
            ...
        ],
        "color": [
            {"rgb": "RGB tuple (r,g,b)", "color_name": "color name"},
            ...
        ],
        "text": [
            {"label": "content of the text", "score": "confidence score"},
            ...
        ],
        "size": [
            {"area": "percentage of the total area", "height": "percentage of the total height", "width": "percentage of the total width"},
            ...
        ]
    }
}
```

### dataset
The dataset includes the query and reasoning path:

```json
{
    "image_id": "image id",
    "url": "URL to download this image",
    "query": "Initial query",
    "reasoning_path": [
        {"sub_question": "Sub-question to be solved in this step", "tool": "Tool used", "operations": "Parameters needed for the tool (optional)"},
        ...
    ]
}
```
