from vllm import LLM, SamplingParams
import json
import argparse
import os


def get_question(case):
    try:
        question = case['prompt'].split('## Question\n')[1].split('\n')[0]
    except:
        question = case['prompt'].split('## Question:\n')[1].split('\n')[0]
    return question

def get_answer(case):
    answer = case['answer']
    if isinstance(answer, list):
        answer = [str(a) for a in answer]
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

def eval_with_llm(benchmark, model_name, llm, sampling_params):
    output_file = f'data/outputs/{model_name}/{benchmark}_llm_eval.json'
    if os.path.exists(output_file) and not args.force_generate:
        print(f'{output_file} already exists, please use --force_generate to overwrite it')
        return 
    
    data = json.load(open(f'data/outputs/{model_name}/{benchmark}_w_sql.json'))

    data = data[:args.end]
    input_prompts = [get_prompt(entry) for entry in data]
    print(input_prompts[0])


    outputs = llm.generate(input_prompts, sampling_params=sampling_params)

    result = []
    for i, output in enumerate(outputs):
        result.append({'prompt': input_prompts[i],
                       'prediction': prediction(data[i]),
                       'answer': data[i]['answer'],
                       'correct':output.outputs[0].text})
        
    json.dump(result, open(output_file, 'w'), indent =4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, required=True)
    parser.add_argument('--llm_path', type = str, required = True)
    parser.add_argument('--force_generate', action = 'store_true')
    parser.add_argument('--benchmark', type = str)
    parser.add_argument('--max_tokens', type = int, default=4)
    parser.add_argument('--tensor_parallel_size', type = int, default=4)
    parser.add_argument('--end', type = int, default=-1)
    args = parser.parse_args()
    
    if args.model_path.endswith('/'):
        model_name = args.model_path.split('/')[-2]
    else:
        model_name = args.model_path.split('/')[-1]


    llm = LLM(args.llm_path, tensor_parallel_size=args.tensor_parallel_size)
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0)
    if args.benchmark is None:
        target_benchmarks = ['hybridqa', 'tatqa', 'wikisql', 'wikitabqa']
    else:
        target_benchmarks = [args.benchmark]
    for benchmark in target_benchmarks:
        eval_with_llm(benchmark, model_name, llm, sampling_params)



