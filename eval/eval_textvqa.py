import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')

    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data_list = [json.loads(_) for _ in f]
    
    score = 0
    for data in data_list:
        text = data['text'].lower()
        for ans in data['answers']:
            if ans.lower() in text:
                score += 1
                break
    print(score / len(data_list))