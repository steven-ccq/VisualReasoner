# VisualReasoner

Official repository for the EMNLP 2024 paper ["From the Least to the Most: Building a Plug-and-Play Visual Reasoner via Data Synthesis"](https://arxiv.org/abs/2406.19934)

![](imgs/pipeline.png)

---

## :gear: Setup

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

Rename the Planner model as `planner` and move it into `models/`.

## :rocket: Inference

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

## :dart: Evaluation

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

## :balloon: Data

We also provide a 1M dataset synthesized using the least-to-most method. You can access this dataset through [VisualReasoner-1M](https://huggingface.co/datasets/orange-sk/VisualReasoner-1M). We also release a variant of this dataset, which contains 30k end-to-end reasoning processes. You can access this dataset through [VisualReasoner-30k](https://huggingface.co/datasets/orange-sk/VisualReasoner-30k).
