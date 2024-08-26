import json

with open('data/GQA/testdev_balanced_questions.json', 'r') as f:
    dataset = json.load(f)

GQA_dataset = []
for id, info in dataset.items():
    data = {
        'image_id': info['imageId'],
        'question': info['question'],
        'answer': info['answer'],
        'full_answer': info['fullAnswer']
    }
    GQA_dataset.append(data)

with open('data/GQA/GQA_testdev_balanced_questions.json', 'w') as f:
    json.dump(GQA_dataset, f)