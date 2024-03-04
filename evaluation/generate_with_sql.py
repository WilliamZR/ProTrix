from vllm import LLM, SamplingParams
import json
import argparse
import os
import multiprocessing
import sqlite3
from tqdm import tqdm
import pandas as pd
import re
from sql_tool import generate_sql_prompt, process_table_datatype, process_table_datatype_transpose, trunacate_input
import warnings
warnings.filterwarnings("ignore")

def run_inference_one_gpu(gpu_id, data, model_name, sampling_params):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = LLM(model_name)
    
    if 'input' in data[0]:
        input_prompts = [d['input'] for d in data]
    else:
        input_prompts = [d['prompt'] for d in data]
    outputs = llm.generate(input_prompts, sampling_params)

    result = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        result.append({ 'id': data[i]['id'],
                        'prompt': prompt, 
                        'generated_text': generated_text,
                        'answer': data[i]['answer']})
                        
    return result


split_list = lambda l, n: [l[i * len(l) // n: (i + 1) * len(l) // n] for i in range(n)]

def run_inference_multi_gpu(model_name, data, sampling_params):

    gpu_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    num_gpus = len(gpu_ids)
    os.environ['WORLD_SIZE'] = str(num_gpus)
    
    print(f'Using {num_gpus} GPUs')
    split_prompts = split_list(data, num_gpus)
    data_splits = [(gpu_ids[i], p, model_name, sampling_params) for i, p in enumerate(split_prompts)]
    print(len(data_splits))
    with multiprocessing.Pool(processes=num_gpus) as pool:
        results = pool.starmap(run_inference_one_gpu, data_splits)

    outputs = []
    for result in results:
        outputs.extend(result)

    return outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, required=True)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--force_generate', action = 'store_true')
    args = parser.parse_args()

    if args.model_path.endswith('/'):
        model_name = args.model_path.split('/')[-2]
    else:
        model_name = args.model_path.split('/')[-1]
    ## check if the output folder exists
    if not os.path.exists(f'data/outputs/{model_name}'):
        os.makedirs(f'data/outputs/{model_name}')

    ## check is the output file exists
    ## if the output file exists, skip generating
    
    output_file = f'data/outputs/{model_name}/{args.benchmark}.json'


    if args.benchmark in ['wikitabqa', 'tabfact', 'wikisql']:
        with open(f'data/evaluation_data/{args.benchmark}_test.json', 'r') as f:
            data = json.load(f)
            print(f'Loading data from data/evaluation_data/{args.benchmark}_test.json')
    else:
        with open(f'data/evaluation_data/{args.benchmark}_dev.json', 'r') as f:
            data = json.load(f)
            print(f'Loading data from data/evaluation_data/{args.benchmark}_dev.json')


    if os.path.exists(output_file) and not args.force_generate:
        print(f'Output file {output_file} exists\n Skip Generating...')
        with open(output_file, 'r') as f:
            result = json.load(f)
    else:
        print(f'Output file {output_file} does not exist\n Generating...')
        data = [trunacate_input(entry) for entry in data]
        result = run_inference_multi_gpu(args.model_path, data, SamplingParams(temperature=0, max_tokens=1024))

        with open(f'data/outputs/{model_name}/{args.benchmark}.json', 'w') as f:
            json.dump(result, f, indent=4)

    output_file_sql = output_file = f'data/outputs/{model_name}/{args.benchmark}_w_sql.json'
    ## get new input with sql
    if os.path.exists(output_file_sql) and not args.force_generate:
        print(f'Output file {output_file_sql} exists\n Skip Generating...')
        exit(0)
    else:
        print(f'Output file {output_file_sql} does not exist\n Generating...')

    print('Converting table data type...Please wait...')
    result_list = split_list(result, 32)
    result_splits = [[d] for d in result_list]
    with multiprocessing.Pool(processes=32) as pool:
        output = pool.starmap(process_table_datatype, result_splits)
    result = []
    for r in output:
        result.extend(r)

    result_list = split_list(result, 32)
    result_splits = [[d] for d in result_list]
    with multiprocessing.Pool(processes=32) as pool:
        output = pool.starmap(process_table_datatype_transpose, result_splits)
    result = []
    for r in output:
        result.extend(r)

    return_after_sql = []

    input_after_sql = []
    for entry in result:
        try:
            entry['prompt'] = generate_sql_prompt(entry)
            input_after_sql.append(entry)
        except:
            return_after_sql.append(entry)

    result_sql = run_inference_multi_gpu(args.model_path, input_after_sql, SamplingParams(temperature=0, max_tokens=1024))
    return_after_sql += result_sql
    
    output = []
    for entry in return_after_sql:
        output.append({'prompt':entry['prompt'],
                       'generated_text':entry['generated_text'],
                        'answer': entry['answer']})
    with open(output_file_sql, 'w') as f:
        json.dump(output, f, indent=4)
    