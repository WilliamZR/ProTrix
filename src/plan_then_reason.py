import openai
import os
import json
import argparse
from tqdm import tqdm
import multiprocessing
import random
import re
openai.api_key = os.getenv("OPENAI_API_KEY")
from prompt.prompt import get_prompt
from evaluation.sql_tool import num_tokens_from_string
import jsonlines
import tiktoken
from table_normalizer import normalize_all_tables

# ONLY use the following code if you need to use a proxy to access the apis.
#os.environ['http_proxy'] = 'http://127.0.0.1:7890'
#os.environ['https_proxy'] = 'http://127.0.0.1:7890'

import warnings
warnings.filterwarnings("ignore")

def add_id(data):
    for i, entry in enumerate(data):
        if 'ids' not in entry:
            entry['ids'] = f'id-{i}'
        else:
            break
    return data

def get_results(data_slice, cpu_id, model_version, output_folder, max_tokens = 512, false_generate = False):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    output_file = os.path.join(output_folder, f'response_{cpu_id}.json')
    if os.path.exists(output_file) and not false_generate:
        print('path already exists! ' + output_file)
        return None
    ret = []


    for i in tqdm(range(len(data_slice)) , desc=f'{cpu_id}'):
        input_text = data_slice[i]['prompt']
        if 'skip' in data_slice[i] and data_slice[i]['skip'] == True:
            ret.append(data_slice[i])
            continue
        if 'SQL failed' in input_text:
            ret.append(data_slice[i])
            ret[i]['output'] = 'Failed SQL'
            continue
        temp = 15
        while temp > 0:
            try:
                completion = openai.chat.completions.create(
                    model = model_version,
                    messages = [{'role' : 'user', 'content' : input_text}],
                    temperature = 0.0,
                    max_tokens = max_tokens,
                    seed=42
                )
                answer = completion.choices[0].message.content
                ret.append(data_slice[i])
                ret[i]['output'] = answer
                break
            except:
                temp -= 1

        if temp > 0:
            continue
        else:
            ret.append(data_slice[i])
            ret[i]['output'] = 'None'

    json.dump(ret, open(output_file, 'w'), indent =4)
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--benchmark', type=str, required=True)
    parser.add_argument('--step', type=str, required=True, help = '[plan, reason, one_step, one_step_result]')
    parser.add_argument('--force_generate', action='store_true')
    parser.add_argument('--end', type=int)
    parser.add_argument('--max_tokens', type = int, default = 1024)
    parser.add_argument('--dry_run', action = 'store_true')
    parser.add_argument('--run', action='store_true', help = 'This helps you to prevent accidentally run all the data before you are ready')
    args = parser.parse_args()

    assert args.step in ['plan', 'reason', 'one_step', 'one_step_result']

    if args.step == 'plan':
        #data = json.load(open(f'data/eval_data_dict/{args.benchmark}.json'))
        ## use jsonlines to load 
        data = []
        with jsonlines.open(f'data/eval_data_dict/{args.benchmark}.jsonl') as f:
            for line in f:
                data.append(line)
    elif args.step == 'reason':
        data = json.load(open(f'data/gpt_output/{args.model_name}/{args.benchmark}_plan_parsed.json'))
    elif args.step == 'one_step':
        data = []
        with jsonlines.open(f'data/eval_data_dict/{args.benchmark}.jsonl') as f:
            for line in f:
                data.append(line) 
    elif args.step == 'one_step_result':
        data = json.load(open(f'data/gpt_output/{args.model_name}/{args.benchmark}_one_step_parsed.json'))
    if args.end:
        data = data[:args.end]
    data = add_id(data)
    data = normalize_all_tables(data)
    ## save normalized table (pd.DataFrame)i
    #print(data[0].keys())
    prompts = get_prompt(args.benchmark, args.step, data)
    print(prompts[4])
    ## compute max tokens
    if args.dry_run:
        max_tokens = max([num_tokens_from_string(prompt) for prompt in prompts])
        print(f'Max tokens: {max_tokens}')
    if args.dry_run or not args.run:
        print('============END OF DRY RUN=================')
        print(data[0]['table_text'])
        exit()
    for entry, prompt in zip(data, prompts):
        entry['prompt'] = prompt

    if not os.path.exists(f'data/gpt_output/{args.model_name}'):
        os.mkdir(f'data/gpt_output/{args.model_name}')

    output_folder = f'data/gpt_output/{args.model_name}/{args.benchmark}_{args.step}'


    itv = 20
    pool = multiprocessing.Pool(16)
    for sid in range(0, len(data), itv):
        pool.apply_async(get_results, args=(data[sid:sid+itv], sid//itv, args.model_name, output_folder, args.max_tokens, args.force_generate))
    pool.close()
    pool.join()

    files = os.listdir(output_folder)
    files = [os.path.join(output_folder, file) for file in files if file.endswith('.json')]
        ## combine all the files into one json
    output_data = []
    for file in files:
        output_data += json.load(open(file))
    assert len(output_data) == len(data)

    output_file = f'data/gpt_output/{args.model_name}/{args.benchmark}_{args.step}.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)