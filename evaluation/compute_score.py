import json
import argparse
import os
import re
from sklearn.metrics import f1_score

def evaluate_tatqa(entry):
    # remove elements following https://github.com/jtonglet/SEER
    output = entry['generated_text']

    pred = output.split('answer is')[-1]
    gold = entry['answer']
    pred = pred.lower().strip()
    pred = pred.replace(',', '')
    pred = pred.replace('"', '')
    elements_to_remove = [",","$","£","'","million","billion","years","%","m","`","(",")","-"]
    for e in elements_to_remove:
        pred = pred.replace(e, '')
    
    if isinstance(gold, list):
        if pred.endswith('.'):
            pred = pred[:-1]
        for g in gold:
            g = str(g).replace(',', '')
            g = g.replace('"', '')
            for e in elements_to_remove:
                g = g.replace(e, '')
            try:
                if re.search(r'{}'.format(str(g).lower().strip()), pred) is  None:
                    return False
            except:
                return False
        return True
    else:
        gold = str(gold).replace(',', '')
        gold = gold.replace('"', '')
        gold = gold.lower().strip()
        for e in elements_to_remove:
            gold = gold.replace(e, '')

        try:
            return re.search(r'{}'.format(str(gold)), pred) is not None
        except:
            return False
        
def evaluate_qa(entry):
    ## FIXME: this heuristic matching method is not perfect, we use llm to evalute performance on QA tasks for now
    ## For llm evaluation, please refer to eval_with_llm.py
    ## You can still use this metric for development, but this metric would generally underestimate the performance
    output = entry['generated_text']
    if 'answer is' in output:
        pred = output.split('answer is')[-1]
    else:
        pred = output.split('\n')[-1]
    gold = entry['answer']
    pred = pred.lower().strip()
    ## check if gold is a list
    if isinstance(gold, list):
       return any(str(g).lower() in pred for g in gold)
    else:
        return str(gold).lower() in pred
    
def evaluate_verification(entry):
    prediction = entry['generated_text']
    if 'answer is' in prediction:
        prediction = prediction.split('answer is')[-1]
    else:
        prediction = prediction.split('\n')[-1]
    answer = entry['answer']
    answer = answer.lower()
    pred_answer = extract_veracity_label(entry, prediction)

    return pred_answer == answer

def extract_veracity_label(entry, prediction):
    
    prediction = prediction.lower()
    if 'not enough info' in prediction:
        pred_answer = 'not enough info'
    else:
        support, refute = False, False
        if 'supports' in prediction or 'supported' in prediction or 'yes' in prediction:
            support = True
        if 'refutes' in prediction or 'refuted' in prediction or re.match('\bno\b',prediction):
            refute = True

        if support and not refute:
            pred_answer = 'supports'
        elif refute and not support:
            pred_answer = 'refutes'
        else:
            pred_answer = 'not enough info'
    return pred_answer



def compute_score_for_llm_evaluation(data):
    correct = len([entry for entry in data  if 'yes' in entry['correct'].lower()])
    return correct / len(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str)
    parser.add_argument('--benchmark', type = str)
    parser.add_argument('--eval_with_llm', action='store_true')

    args = parser.parse_args()

    if args.model_path.endswith('/'):
        model_name = args.model_path.split('/')[-2]
    else:
        model_name = args.model_path.split('/')[-1]

    if args.eval_with_llm:
        with open(f'data/outputs/{model_name}/{args.benchmark}_llm_eval.json', 'r') as f:
            outputs = json.load(f)
        print('llm result')
        accuracy = compute_score_for_llm_evaluation(outputs)
        print('Accuracy: {}'.format(accuracy))
        exit(0)
    try:
        with open(f'data/outputs/{model_name}/{args.benchmark}_w_sql.json', 'r') as f:
            outputs = json.load(f)
    except:
        with open(f'data/outputs/{model_name}/{args.benchmark}.json', 'r') as f:
            outputs = json.load(f)
    if args.benchmark in ['wikisql', 'wikitabqa', 'hybridqa', 'finqa']:
        accuracy = sum([evaluate_qa(entry) for entry in outputs]) / len(outputs)
        print('Accuracy: {}'.format(accuracy))
    elif args.benchmark in ['tabfact', 'feverous']:
        accuracy = sum([evaluate_verification(entry) for entry in outputs]) / len(outputs)
        print('Accuracy: {}'.format(accuracy))
    elif args.benchmark == 'tatqa':
        accuracy = sum([evaluate_tatqa(entry) for entry in outputs]) / len(outputs)
        print('Accuracy: {}'.format(accuracy))
    elif args.benchmark == 'scitab':
        label_dict = {'supports': 0, 'refutes': 1, 'not enough info': 2}
        pred_answers = [extract_veracity_label(entry, entry['generated_text']) for entry in outputs]
        pred_answers = [label_dict[entry] for entry in pred_answers]
        gold_answers = [entry['answer'] for entry in outputs]
        gold_answers = [label_dict[entry] for entry in gold_answers]
        f1 = f1_score(gold_answers, pred_answers, average='macro')
        print('F1: {}'.format(f1))
    
    if os.path.exists(f'data/outputs/{model_name}/result.json'):
        with open(f'data/outputs/{model_name}/result.json', 'r') as f:
            result = json.load(f)
    else:
        result = {}

    result[args.benchmark] = accuracy

    with open(f'data/outputs/{model_name}/result.json', 'w') as writer:
        json.dump(result, writer, indent=4)