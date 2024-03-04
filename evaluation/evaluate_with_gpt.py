import openai
import os
import json
import argparse
from tqdm import tqdm
import multiprocessing
import random
import re
openai.api_key = os.getenv("OPENAI_API_KEY")

## you can modify or delete the following lines if you don't need proxy
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

def get_results(data_slice, cpu_id, model_version, output_folder, max_tokens = 4):
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
                ret[i]['correct'] = answer
                break
            except:
                temp -= 1

        if temp > 0:
            continue
        else:
            ret.append(data_slice[i])
            ret[i]['correct'] = 'None'

    json.dump(ret, open(output_file, 'w'), indent =4)
    return None


def get_question(case):
    try:
        question = case['prompt'].split('## Claim\n')[1].split('\n')[0]
    except:
        question = case['prompt'].split('## Question\n')[1].split('\n')[0]
    return question

def get_answer(case):
    answer = case['answer']
    answer = [str(a) for a in answer]
    if isinstance(answer, list):
        answer = '\n'.join(answer)
    return answer

def prediction(case):
    try:
        prediction = 'The predicted answer is' + case['generated_text'].split('answer is')[-1]
    except:
        prediction = case['generated_text'].split('\n')[-1]
    return prediction

def get_prompt(case):
    prompt = 'Check if the prediction answers the question correctly. For numerical answers, you should check if the predicted answer is approximately correct. For questions with multiple answers, you should check if all the predicted answers are correct. If the predicted answer is correct, return "Yes". Otherwise, return "No". The question, predicted answer, and gold answer are provided below.'
    prompt += f'\n\nQuestion:\n{get_question(case)}'
    prompt += f'\n\nGold Answer:\n{get_answer(case)}'
    prompt += f'\n\nPredicted Answer:\n{prediction(case)}'
    prompt += f'\n\nDoes the predicted answer the question correctly? Yes/No\n\n'
    prompt += 'Answer:'
    return prompt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--force_generate', action='store_true', help = 'overwrite all the existing files')
    parser.add_argument('--debug', action='store_true', help = 'only generate 20 responses')
    parser.add_argument('--dry_run', action='store_true', help = 'only print the prompt')
    parser.add_argument('--end', type = int, default = 1000)
    args = parser.parse_args()

    data = json.load(open(f'../data/outputs/{args.model_name}/{args.benchmark}_w_sql.json'))
    print('loaded from file' + f'../data/outputs/{args.model_name}/{args.benchmark}_w_sql.json')
    data = data[:args.end]
    print(f'Running ChatGPT evaluation on {len(data)} cases')

    output_folder = f'../data/outputs/{args.model_name}/{args.benchmark}_evaluation'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    output_file = f'../data/outputs/{args.model_name}/{args.benchmark}_evaluation_gpt.json'
    # check if the final output is already generated
    if os.path.exists(output_file) and not args.force_generate:
        print(f'{output_file} already exists. Skipping...')
        exit()

    for entry in data:
        entry['prompt'] = get_prompt(entry)
    print(data[0]['prompt'])
    if args.dry_run:
        exit(0)
    itv = 20
    if args.debug:
        output = get_results(data[:itv], 0, 'gpt-3.5-turbo', output_folder)
    else:
        pool = multiprocessing.Pool(processes=16)
        for sidin in range(0, len(data), itv):
            pool.apply_async(get_results, (data[sidin:sidin+itv], sidin, 'gpt-3.5-turbo', output_folder))
        pool.close()
        pool.join()

        files = [f'{output_folder}/response_{i*itv}.json' for i in range(len(data)//itv )]
        ## combine all the files into one json
        output_data = []
        for file in files:
            output_data += json.load(open(file))
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)

