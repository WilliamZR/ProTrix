import openai
import os
import json
import argparse
from tqdm import tqdm
import multiprocessing
import random
import re
openai.api_key = os.getenv("OPENAI_API_KEY")

# ONLY use the following code if you need to use a proxy to access the apis.
#os.environ['http_proxy'] = 'http://127.0.0.1:7890'
#os.environ['https_proxy'] = 'http://127.0.0.1:7890'

def get_results(data_slice, cpu_id, model_version, output_folder, max_tokens = 16):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    output_file = os.path.join(output_folder, f'response_{cpu_id}.json')
    if os.path.exists(output_file):
        print('path already exists! ' + output_file)
        return None
    ret = []


    for i in tqdm(range(len(data_slice)) , desc=f'{cpu_id}'):
        input_text = data_slice[i]['prompt']
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
    parser.add_argument('--reasoning_type', type=str, required=True)
    parser.add_argument('--force_generate', action='store_true')
    parser.add_argument('--debug', type = int,required=True)
    parser.add_argument('--max_tokens', type = int, default = 16)
    args = parser.parse_args()
    if args.debug == 0:
        args.debug = False
    else:
        args.debug = True
    print(args)
    if not os.path.exists(f'../data/outputs/{args.model_name}'):
        os.mkdir(f'../data/outputs/{args.model_name}')

    output_file = f'../data/outputs/{args.model_name}/{args.benchmark}_{args.reasoning_type}.json'
    if os.path.exists(output_file) and not args.force_generate:
        print(f'{output_file} already exists. Skipping...')
        exit()
    data = json.load(open(f'../data/evaluation_data/end2end/{args.benchmark}.json'))

    input_prompts = [entry['prompt'] for entry in data]
    print('==============PROMPT======================')
    print(input_prompts[0])
    print('=======================================')

    itv = 20
    output_folder = f'../data/outputs/{args.model_name}/{args.benchmark}_{args.reasoning_type}_response'

    if args.debug:
        output = get_results(data[:itv], 0, args.model_name, output_folder)
    else:
        pool = multiprocessing.Pool(16)
        for sid in range(0, len(data), itv):
            pool.apply_async(get_results, args=(data[sid:sid+itv], sid//itv, args.model_name, output_folder, args.max_tokens))
        pool.close()
        pool.join()

        files = os.listdir(output_folder)
        files = [os.path.join(output_folder, file) for file in files if file.endswith('.json')]
        ## combine all the files into one json
        output_data = []
        for file in files:
            output_data += json.load(open(file))
        assert len(output_data) == len(data)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)