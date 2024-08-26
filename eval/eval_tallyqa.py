import json
import num2words
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')

    args = parser.parse_args()

    with open(args.input, 'r') as f:
        eval_dataset = [json.loads(_) for _ in f]

    simple_num = 0
    simple_match = 0
    complex_num = 0
    complex_match = 0
    for data in eval_dataset:
        if data['issimple']:
            simple_num += 1
        else:
            complex_num += 1
        if str(data['answer']) in data['response'].split() or num2words.num2words(data['answer']) in data['response']:
            if data['issimple']:
                simple_match += 1
            else:
                complex_match += 1

    print('simple: {}'.format(simple_match/simple_num))
    print('complex: {}'.format(complex_match/complex_num))